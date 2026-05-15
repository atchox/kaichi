use super::AssignmentModel;
use super::em::{clamp_probability, log_binom_pmf, logsumexp2, run_em, sigmoid, solve_2x2_sym};
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};

/// Per-guide Binomial mixture using total guide UMIs as the number of trials.
///
/// logit(p_i) = β0 + β1·z_i. Trials n_i = total_counts_i (total guide UMIs per cell).
/// M-step uses logistic IRLS (Fisher scoring). Cells with n_i = 0 are excluded.
pub struct BinomialModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub inner_max_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
}

impl Default for BinomialModel {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            max_em_iters: 100,
            inner_max_iters: 25,
            tol: 1e-6,
            min_nonzero: 2,
            min_max_count: 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FitParams {
    pi: f64,
    beta0: f64,
    beta1: f64,
}

impl AssignmentModel for BinomialModel {
    fn name(&self) -> &'static str {
        "binomial"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();
        let total_counts = &input.covariates.total_counts;

        // Per-cell trial count (total guide UMIs).
        let trial_counts: Vec<f64> = (0..n_cells)
            .map(|i| total_counts.value(i) as f64)
            .collect();

        let guide_fits: Vec<Option<FitParams>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                // Data: (y_i, n_i) where y_i = guide count, n_i = total guide UMIs.
                let data: Vec<(f64, f64)> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&i, &v)| v > 0 && trial_counts[i] > 0.0)
                    .map(|(&i, &v)| (v as f64, trial_counts[i]))
                    .collect();
                self.fit_guide(&data)
            })
            .collect();

        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let n_i = trial_counts[cell];
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                let Some(params) = guide_fits[guide_idx] else { continue };
                if n_i <= 0.0 { continue; }
                let post = posterior_signal(count as f64, n_i, params) as f32;
                if post >= self.min_confidence {
                    n_passing += 1;
                    match best {
                        None => best = Some((guide_idx, post, count)),
                        Some((_, bp, bc)) => {
                            if post > bp || (post == bp && count > bc) {
                                best = Some((guide_idx, post, count));
                            }
                        }
                    }
                    out.push_assigned_triple(cell, guide_idx);
                }
            }

            let n_det = n_detected_u8(n_passing);
            match best {
                None => out.append_unassigned(n_det),
                Some((guide_idx, post, count)) => {
                    out.append_assigned(
                        guide_ids_arr.value(guide_idx),
                        count,
                        post,
                        n_passing > 1,
                        n_det,
                    );
                }
            }
        }

        out.finish(cell_barcodes, true)
    }

    fn params_json(&self) -> Value {
        json!({
            "min_confidence": self.min_confidence,
            "max_em_iters": self.max_em_iters,
            "inner_max_iters": self.inner_max_iters,
            "tol": self.tol,
            "min_nonzero": self.min_nonzero,
            "min_max_count": self.min_max_count,
        })
    }
}

impl BinomialModel {
    fn fit_guide(&self, data: &[(f64, f64)]) -> Option<FitParams> {
        let n_nonzero = data.len() as u32;
        let max_count = data.iter().map(|(y, _)| *y as u32).max().unwrap_or(0);
        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }
        Some(fit_mixture(data, self.max_em_iters, self.inner_max_iters, self.tol as f64))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

fn fit_mixture(data: &[(f64, f64)], max_em_iters: u32, inner_max_iters: u32, tol: f64) -> FitParams {
    let init = initialize_params(data);
    let mut responsibilities = vec![0.0f64; data.len()];

    run_em(
        init,
        |params| {
            let p0 = sigmoid(params.beta0);
            let p1 = sigmoid(params.beta0 + params.beta1);
            let mut log_lik = 0.0;
            for (idx, &(y, n)) in data.iter().enumerate() {
                let log_bg = (1.0 - params.pi).ln() + log_binom_pmf(y, n, p0);
                let log_sig = params.pi.ln() + log_binom_pmf(y, n, p1);
                let denom = logsumexp2(log_bg, log_sig);
                responsibilities[idx] = (log_sig - denom).exp();
                log_lik += denom;
            }
            let new_params = m_step(params, data, &responsibilities, inner_max_iters);
            (new_params, log_lik)
        },
        max_em_iters,
        tol,
    )
}

fn initialize_params(data: &[(f64, f64)]) -> FitParams {
    // Initialize with background p ≈ mean(y/n) and signal p higher.
    let mean_prop: f64 = data.iter().map(|(y, n)| y / n).sum::<f64>() / data.len() as f64;
    let max_prop: f64 = data.iter().map(|(y, n)| y / n).fold(0.0_f64, f64::max);
    let beta0 = (mean_prop / (1.0 - mean_prop).max(1e-6)).ln().clamp(-5.0, 5.0);
    let beta1 = ((max_prop / (1.0 - max_prop).max(1e-6)).ln() - beta0).max(0.5);
    FitParams { pi: 0.1, beta0, beta1 }
}

fn m_step(params: FitParams, data: &[(f64, f64)], resp: &[f64], inner_max_iters: u32) -> FitParams {
    let pi = clamp_probability(resp.iter().sum::<f64>() / data.len() as f64);
    let mut beta0 = params.beta0;
    let mut beta1 = params.beta1;

    for _ in 0..inner_max_iters {
        let p0 = sigmoid(beta0);
        let p1 = sigmoid(beta0 + beta1);

        let mut g = [0.0f64; 2];
        let mut fi = [0.0f64; 3]; // Fisher info: [I00, I01, I11]
        for (&(y, n), &r) in data.iter().zip(resp) {
            // Score ∂log_binom/∂β = n·(y/n - p)·∂p/∂β, ∂p/∂β = p(1-p) for logit link
            g[0] += (1.0 - r) * (y - n * p0) + r * (y - n * p1);
            g[1] += r * (y - n * p1);
            // Fisher info (working weight for binomial logistic): n·p·(1-p)
            fi[0] += (1.0 - r) * n * p0 * (1.0 - p0) + r * n * p1 * (1.0 - p1);
            fi[1] += r * n * p1 * (1.0 - p1);
            fi[2] += r * n * p1 * (1.0 - p1);
        }
        let Some(delta) = solve_2x2_sym(fi, g) else { break };
        let d0 = delta[0].clamp(-3.0, 3.0);
        let d1 = delta[1].clamp(-3.0, 3.0);
        beta0 += d0;
        beta1 += d1;
        if d0.abs() < 1e-8 && d1.abs() < 1e-8 { break; }
    }

    if beta1 < 0.0 { beta1 = -beta1; }

    FitParams { pi, beta0, beta1 }
}

fn posterior_signal(y: f64, n: f64, params: FitParams) -> f64 {
    let p0 = sigmoid(params.beta0);
    let p1 = sigmoid(params.beta0 + params.beta1);
    let log_bg = (1.0 - params.pi).ln() + log_binom_pmf(y, n, p0);
    let log_sig = params.pi.ln() + log_binom_pmf(y, n, p1);
    let denom = logsumexp2(log_bg, log_sig);
    (log_sig - denom).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_support::{input_with_row_sums as make_input, input_with_totals};
    use arrow::array::BooleanArray;

    fn make_input_with_totals(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        totals: Vec<u32>,
    ) -> LoadedInput {
        input_with_totals(n_cells, n_guides, triples,
            totals.iter().map(|&t| t as f32).collect())
    }

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    #[test]
    fn assigns_high_proportion_cells() {
        // Background: y=1 out of n=50 (proportion=0.02).
        // Signal: y=40 out of n=50 (proportion=0.80).
        let input = make_input_with_totals(
            6, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,40),(4,0,42),(5,0,45)],
            vec![50,50,50,50,50,50],
        );
        let model = BinomialModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        assert!(is_u.value(0));
        assert!(is_u.value(1));
        assert!(is_u.value(2));
        assert!(!is_u.value(3));
        assert!(!is_u.value(4));
        assert!(!is_u.value(5));
    }

    #[test]
    fn skips_guides_with_too_little_signal() {
        let input = make_input_with_totals(
            3, 1,
            vec![(1,0,1),(2,0,1)],
            vec![50,50,50],
        );
        let result = BinomialModel::default().assign(&input).unwrap();
        for i in 0..3 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = BinomialModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        let input = make_input_with_totals(3, 1, vec![(1,0,1),(2,0,1)], vec![50,50,50]);
        let result = BinomialModel::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
            assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(i));
        }
    }

    #[test]
    fn multi_infected_flagged() {
        // Cell 3 has signal-level counts for both guides. totals=100 so bg prop=0.01, signal prop≈0.40.
        let input = make_input_with_totals(8, 2, vec![
            (0,0,1),(0,1,1),(1,0,1),(1,1,1),(2,0,1),
            (3,0,40),(3,1,42),
            (4,0,41),(5,0,40),
            (6,1,43),(7,1,41),
        ], vec![100; 8]);
        let model = BinomialModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(3), "cell 3 should be multi-infected");
    }

    // ---- posterior_signal (Binomial Bayes) ----

    #[test]
    fn binomial_posterior_equals_pi_when_components_identical() {
        // β1 = 0 ⇒ p_signal = p_bg ⇒ likelihoods identical ⇒ posterior = π.
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = FitParams { pi, beta0: -2.0, beta1: 0.0 };
            let post = posterior_signal(5.0, 50.0, p);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn binomial_posterior_monotonic_in_y_when_beta1_positive() {
        // logit(p_bg) = -3, logit(p_sig) = 1 ⇒ p_bg ≈ 0.047, p_sig ≈ 0.73.
        // Saturates rapidly under f64; test non-decreasing globally and strict
        // monotonicity in the un-saturated low-y range.
        let p = FitParams { pi: 0.3, beta0: -3.0, beta1: 4.0 };
        let n = 50.0;
        let ys = [0.0, 1.0, 5.0, 10.0, 15.0, 30.0, 45.0];
        let posts: Vec<f64> = ys.iter().map(|&y| posterior_signal(y, n, p)).collect();
        for w in posts.windows(2) {
            assert!(w[1] >= w[0], "posterior decreased: {} → {}", w[0], w[1]);
        }
        // Strict increase over the un-saturated stretch (y = 0..10).
        assert!(posts[0] < posts[3], "no strict increase 0 → 10: {} → {}", posts[0], posts[3]);
    }

    #[test]
    fn binomial_posterior_saturates() {
        // p_bg ≈ 0.047, p_sig ≈ 0.73; y=45/50 is overwhelmingly signal, y=0/50 overwhelmingly bg.
        let p = FitParams { pi: 0.2, beta0: -3.0, beta1: 4.0 };
        assert!(posterior_signal(45.0, 50.0, p) > 0.99);
        assert!(posterior_signal(0.0, 50.0, p) < 0.2);
    }

    // ---- fit_mixture: synthetic recovery ----

    #[test]
    fn fit_mixture_recovers_bimodal_binomial() {
        // 12 background (1–2 of 50, prop ≈ 0.03) + 8 signal (38–45 of 50, prop ≈ 0.83).
        let n = 50.0;
        let mut data: Vec<(f64, f64)> = (0..12).map(|i| (1.0 + (i % 2) as f64, n)).collect();
        data.extend((0..8).map(|i| (38.0 + i as f64, n)));

        let fitted = fit_mixture(&data, 200, 25, 1e-8);
        let p_bg = sigmoid(fitted.beta0);
        let p_sig = sigmoid(fitted.beta0 + fitted.beta1);

        assert!(p_sig > p_bg, "p_sig={p_sig} ≤ p_bg={p_bg}");
        assert!(p_bg < 0.10, "p_bg should track low cluster (~0.03), got {p_bg}");
        assert!(p_sig > 0.70, "p_sig should track high cluster (~0.83), got {p_sig}");
        assert!((0.2..=0.6).contains(&fitted.pi), "pi: {}", fitted.pi);
    }
}

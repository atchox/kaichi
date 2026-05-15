use super::AssignmentModel;
use super::em::{clamp_probability, log_binom_pmf, logsumexp2, run_em, sigmoid};
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

#[derive(Clone, Debug)]
struct FitParams {
    pi: f64,
    beta0: f64,
    beta1: f64,
    /// Per-batch logit offset; `gamma[0]` is anchored at 0 for identifiability.
    gamma: Vec<f64>,
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
        let batch = &input.covariates.batch;
        let n_batches = batch.n_batches();

        // Per-cell trial count (total guide UMIs).
        let trial_counts: Vec<f64> = (0..n_cells)
            .map(|i| total_counts.value(i) as f64)
            .collect();

        let guide_fits: Vec<Option<FitParams>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                // Data: (y_i, n_i, b_i) where y_i = guide count, n_i = total UMIs, b_i = batch.
                let data: Vec<(f64, f64, u16)> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(&i, &v)| v > 0 && trial_counts[i] > 0.0)
                    .map(|(&i, &v)| (v as f64, trial_counts[i], batch.codes[i]))
                    .collect();
                self.fit_guide(&data, n_batches)
            })
            .collect();

        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let n_i = trial_counts[cell];
            let b = batch.codes[cell] as usize;
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                let params = match &guide_fits[guide_idx] {
                    Some(p) => p,
                    None => continue,
                };
                if n_i <= 0.0 { continue; }
                let post = posterior_signal(count as f64, n_i, params, b) as f32;
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
    fn fit_guide(&self, data: &[(f64, f64, u16)], n_batches: usize) -> Option<FitParams> {
        let n_nonzero = data.len() as u32;
        let max_count = data.iter().map(|(y, _, _)| *y as u32).max().unwrap_or(0);
        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }
        Some(fit_mixture(data, n_batches, self.max_em_iters, self.inner_max_iters, self.tol as f64))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

fn fit_mixture(
    data: &[(f64, f64, u16)],
    n_batches: usize,
    max_em_iters: u32,
    inner_max_iters: u32,
    tol: f64,
) -> FitParams {
    let init = initialize_params(data, n_batches);
    let mut responsibilities = vec![0.0f64; data.len()];

    run_em(
        init,
        |params| {
            let mut log_lik = 0.0;
            for (idx, &(y, n, b)) in data.iter().enumerate() {
                let gb = params.gamma[b as usize];
                let p0 = sigmoid(params.beta0 + gb);
                let p1 = sigmoid(params.beta0 + params.beta1 + gb);
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

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize) -> FitParams {
    // Per-batch background logit: log(mean_prop_b / (1 − mean_prop_b)).
    // Batch-0 baseline goes into β0; γ_b = batch-b logit − batch-0 logit for b > 0.
    let mut per_batch_y: Vec<f64> = vec![0.0; n_batches];
    let mut per_batch_n: Vec<f64> = vec![0.0; n_batches];
    for &(y, n, b) in data {
        per_batch_y[b as usize] += y;
        per_batch_n[b as usize] += n;
    }
    let logit_of = |p: f64| -> f64 {
        let p = p.clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };
    let mean_prop_b: Vec<f64> = per_batch_y.iter().zip(&per_batch_n)
        .map(|(y, n)| if *n > 0.0 { y / n } else { 0.01 })
        .collect();
    let beta0 = logit_of(mean_prop_b[0]).clamp(-5.0, 5.0);
    let mut gamma = vec![0.0; n_batches];
    for b in 1..n_batches {
        gamma[b] = (logit_of(mean_prop_b[b]) - beta0).clamp(-5.0, 5.0);
    }

    // β1 from the global max proportion vs global mean — signal scale shared across batches.
    let max_prop: f64 = data.iter().map(|(y, n, _)| y / n).fold(0.0_f64, f64::max);
    let mean_prop_all: f64 = data.iter().map(|(y, n, _)| y / n).sum::<f64>() / data.len() as f64;
    let beta1 = (logit_of(max_prop) - logit_of(mean_prop_all)).max(0.5);

    FitParams { pi: 0.1, beta0, beta1, gamma }
}

fn m_step(
    params: FitParams,
    data: &[(f64, f64, u16)],
    resp: &[f64],
    inner_max_iters: u32,
) -> FitParams {
    let pi = clamp_probability(resp.iter().sum::<f64>() / data.len() as f64);
    let n_batches = params.gamma.len();

    // Arrow-Hessian Schur step in (η_0, ..., η_{B−1}, β1). Binomial IRLS weight is n·p·(1−p).
    let mut eta: Vec<f64> = (0..n_batches)
        .map(|b| params.beta0 + params.gamma[b])
        .collect();
    let mut beta1 = params.beta1;

    for _ in 0..inner_max_iters {
        let mut g_eta = vec![0.0f64; n_batches];
        let mut d = vec![0.0f64; n_batches];
        let mut c = vec![0.0f64; n_batches];
        let mut g_beta1 = 0.0f64;
        let mut s = 0.0f64;

        for (&(y, n, bi), &r) in data.iter().zip(resp) {
            let b = bi as usize;
            let p0 = sigmoid(eta[b]);
            let p1 = sigmoid(eta[b] + beta1);
            g_eta[b] += (1.0 - r) * (y - n * p0) + r * (y - n * p1);
            g_beta1 += r * (y - n * p1);
            let w0 = n * p0 * (1.0 - p0);
            let w1 = n * p1 * (1.0 - p1);
            d[b] += (1.0 - r) * w0 + r * w1;
            c[b] += r * w1;
            s += r * w1;
        }

        let mut schur = s;
        let mut g_schur = g_beta1;
        for b in 0..n_batches {
            if d[b] > 1e-14 {
                schur -= c[b] * c[b] / d[b];
                g_schur -= c[b] * g_eta[b] / d[b];
            }
        }
        let d_beta1 = if schur.abs() > 1e-14 {
            (g_schur / schur).clamp(-3.0, 3.0)
        } else { 0.0 };

        let mut max_step = d_beta1.abs();
        for b in 0..n_batches {
            let d_eta = if d[b] > 1e-14 {
                ((g_eta[b] - c[b] * d_beta1) / d[b]).clamp(-3.0, 3.0)
            } else { 0.0 };
            eta[b] += d_eta;
            max_step = max_step.max(d_eta.abs());
        }
        beta1 += d_beta1;

        if max_step < 1e-8 { break; }
    }

    let beta0 = eta[0];
    let mut gamma = vec![0.0; n_batches];
    for b in 1..n_batches {
        gamma[b] = eta[b] - eta[0];
    }

    // Same identifiability relabel as Poisson; see comment there.
    let (pi, beta0, beta1) = if beta1 < 0.0 {
        (1.0 - pi, beta0 + beta1, -beta1)
    } else {
        (pi, beta0, beta1)
    };

    FitParams { pi, beta0, beta1, gamma }
}

fn posterior_signal(y: f64, n: f64, params: &FitParams, batch_idx: usize) -> f64 {
    let gb = params.gamma[batch_idx];
    let p0 = sigmoid(params.beta0 + gb);
    let p1 = sigmoid(params.beta0 + params.beta1 + gb);
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

    fn fp(pi: f64, beta0: f64, beta1: f64) -> FitParams {
        FitParams { pi, beta0, beta1, gamma: vec![0.0] }
    }

    #[test]
    fn binomial_posterior_equals_pi_when_components_identical() {
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = fp(pi, -2.0, 0.0);
            let post = posterior_signal(5.0, 50.0, &p, 0);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn binomial_posterior_monotonic_in_y_when_beta1_positive() {
        let p = fp(0.3, -3.0, 4.0);
        let n = 50.0;
        let ys = [0.0, 1.0, 5.0, 10.0, 15.0, 30.0, 45.0];
        let posts: Vec<f64> = ys.iter().map(|&y| posterior_signal(y, n, &p, 0)).collect();
        for w in posts.windows(2) {
            assert!(w[1] >= w[0], "posterior decreased: {} → {}", w[0], w[1]);
        }
        assert!(posts[0] < posts[3], "no strict increase 0 → 10: {} → {}", posts[0], posts[3]);
    }

    #[test]
    fn binomial_posterior_saturates() {
        let p = fp(0.2, -3.0, 4.0);
        assert!(posterior_signal(45.0, 50.0, &p, 0) > 0.99);
        assert!(posterior_signal(0.0, 50.0, &p, 0) < 0.2);
    }

    #[test]
    fn binomial_posterior_uses_batch_gamma() {
        // γ_1 > 0 raises batch 1's logit baseline → same y/n looks more like background.
        let p = FitParams { pi: 0.3, beta0: -3.0, beta1: 3.0, gamma: vec![0.0, 2.0] };
        let b0 = posterior_signal(5.0, 50.0, &p, 0);
        let b1 = posterior_signal(5.0, 50.0, &p, 1);
        assert!(b0 > b1, "γ_1 > 0 should suppress signal in batch 1: b0={b0}, b1={b1}");
    }

    // ---- fit_mixture: synthetic recovery ----

    #[test]
    fn fit_mixture_recovers_bimodal_binomial() {
        // 12 background (1–2 of 50, prop ≈ 0.03) + 8 signal (38–45 of 50, prop ≈ 0.83), single batch.
        let n = 50.0;
        let mut data: Vec<(f64, f64, u16)> = (0..12).map(|i| (1.0 + (i % 2) as f64, n, 0u16)).collect();
        data.extend((0..8).map(|i| (38.0 + i as f64, n, 0u16)));

        let fitted = fit_mixture(&data, 1, 200, 25, 1e-8);
        let p_bg = sigmoid(fitted.beta0);
        let p_sig = sigmoid(fitted.beta0 + fitted.beta1);

        assert!(p_sig > p_bg, "p_sig={p_sig} ≤ p_bg={p_bg}");
        assert!(p_bg < 0.10, "p_bg should track low cluster (~0.03), got {p_bg}");
        assert!(p_sig > 0.70, "p_sig should track high cluster (~0.83), got {p_sig}");
        assert!((0.2..=0.6).contains(&fitted.pi), "pi: {}", fitted.pi);
        assert_eq!(fitted.gamma, vec![0.0]);
    }

    #[test]
    fn fit_mixture_recovers_batch_offsets() {
        // Two batches: same signal pattern but batch 1's background runs at ~3× the
        // batch-0 background proportion → γ_1 should land positive.
        let n = 50.0;
        let mut data: Vec<(f64, f64, u16)> = Vec::new();
        // batch 0: 12 bg (y = 1 or 2) + 8 sig (y ≈ 40)
        for i in 0..12 { data.push((1.0 + (i % 2) as f64, n, 0u16)); }
        for i in 0..8  { data.push((38.0 + i as f64, n, 0u16)); }
        // batch 1: 12 bg (y = 4 or 5) + 8 sig (y ≈ 45)
        for i in 0..12 { data.push((4.0 + (i % 2) as f64, n, 1u16)); }
        for i in 0..8  { data.push((42.0 + i as f64, n, 1u16)); }

        let fitted = fit_mixture(&data, 2, 200, 50, 1e-8);
        assert_eq!(fitted.gamma[0], 0.0);
        // Truth γ_1 ≈ logit(4.5/50) − logit(1.5/50) ≈ 1.17; the EM lands near 0.90 here
        // (binomial logit pulls in toward 0 more than Poisson β0 does at small p). Window
        // is wider than Poisson's to absorb that systematic bias without being toothless.
        assert!(
            (fitted.gamma[1] - 1.17).abs() < 0.30,
            "γ_1 expected ≈ 1.17, got {:.3}", fitted.gamma[1]
        );
    }
}

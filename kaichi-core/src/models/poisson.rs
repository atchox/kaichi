use super::AssignmentModel;
use super::em::{clamp_probability, log_poisson_pmf, logsumexp2, run_em, solve_2x2_sym};
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};

/// Per-guide Poisson mixture with depth covariate.
///
/// Background and signal both follow Poisson(μ_i) where
/// log(μ_i) = β0 + β1·z_i + log(total_counts_i + 1).
/// Single-batch; γ_b support can be added when the I/O layer provides batch labels.
pub struct PoissonModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub inner_max_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
}

impl Default for PoissonModel {
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

impl AssignmentModel for PoissonModel {
    fn name(&self) -> &'static str {
        "poisson"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();
        let total_counts = &input.covariates.total_counts;

        let log_depths: Vec<f64> = (0..n_cells)
            .map(|i| (total_counts.value(i) as f64 + 1.0).ln())
            .collect();

        let guide_fits: Vec<Option<FitParams>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let data: Vec<(f64, f64)> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(_, &v)| v > 0)
                    .map(|(&i, &v)| (v as f64, log_depths[i]))
                    .collect();
                self.fit_guide(&data)
            })
            .collect();

        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let log_d = log_depths[cell];
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                let Some(params) = guide_fits[guide_idx] else { continue };
                let post = posterior_signal(count as f64, log_d, params) as f32;
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

impl PoissonModel {
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
            let mut log_lik = 0.0;
            for (idx, &(y, log_d)) in data.iter().enumerate() {
                let mu0 = (params.beta0 + log_d).exp();
                let mu1 = (params.beta0 + params.beta1 + log_d).exp();
                let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(y, mu0);
                let log_sig = params.pi.ln() + log_poisson_pmf(y, mu1);
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
    let max_y: f64 = data.iter().map(|(y, _)| *y).fold(f64::NEG_INFINITY, f64::max);
    let mean_depth: f64 = data.iter().map(|(_, d)| d.exp()).sum::<f64>() / data.len() as f64;
    let mut sorted_y: Vec<f64> = data.iter().map(|(y, _)| *y).collect();
    sorted_y.sort_by(f64::total_cmp);
    let median_y = sorted_y[sorted_y.len() / 2].max(1.0);
    let beta0 = (median_y / mean_depth.max(1e-6)).ln().clamp(-10.0, 10.0);
    let beta1 = (max_y / median_y).ln().max(0.1);
    FitParams { pi: 0.1, beta0, beta1 }
}

fn m_step(params: FitParams, data: &[(f64, f64)], resp: &[f64], inner_max_iters: u32) -> FitParams {
    let pi = clamp_probability(resp.iter().sum::<f64>() / data.len() as f64);
    let mut beta0 = params.beta0;
    let mut beta1 = params.beta1;

    for _ in 0..inner_max_iters {
        let mut g = [0.0f64; 2];
        let mut fi = [0.0f64; 3]; // Fisher info: [I00, I01, I11]
        for (&(y, log_d), &r) in data.iter().zip(resp) {
            let mu0 = (beta0 + log_d).exp();
            let mu1 = (beta0 + beta1 + log_d).exp();
            g[0] += (1.0 - r) * (y - mu0) + r * (y - mu1);
            g[1] += r * (y - mu1);
            fi[0] += (1.0 - r) * mu0 + r * mu1;
            fi[1] += r * mu1;
            fi[2] += r * mu1;
        }
        let Some(delta) = solve_2x2_sym(fi, g) else { break };
        let d0 = delta[0].clamp(-3.0, 3.0);
        let d1 = delta[1].clamp(-3.0, 3.0);
        beta0 += d0;
        beta1 += d1;
        if d0.abs() < 1e-8 && d1.abs() < 1e-8 { break; }
    }

    // Ensure beta1 > 0 (signal must have larger mean than background).
    if beta1 < 0.0 { beta1 = -beta1; }

    FitParams { pi, beta0, beta1 }
}

fn posterior_signal(y: f64, log_d: f64, params: FitParams) -> f64 {
    let mu0 = (params.beta0 + log_d).exp();
    let mu1 = (params.beta0 + params.beta1 + log_d).exp();
    let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(y, mu0);
    let log_sig = params.pi.ln() + log_poisson_pmf(y, mu1);
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
    fn assigns_high_signal_cells() {
        // Background: y=1 of 50. Signal: y=20-25 of 50. Uniform depth → 1D Poisson mixture.
        let input = make_input_with_totals(
            6, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,20),(4,0,22),(5,0,25)],
            vec![50,50,50,50,50,50],
        );
        let model = PoissonModel { min_confidence: 0.8, ..Default::default() };
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
        let input = make_input(3, 1, vec![(1,0,1),(2,0,1)]);
        let result = PoissonModel::default().assign(&input).unwrap();
        for i in 0..3 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = PoissonModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        let input = make_input(3, 1, vec![(1,0,1),(2,0,1)]);
        let result = PoissonModel::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
            assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(i));
        }
    }

    // TODO: test that depth covariate actually affects assignment — e.g. same raw count at 2×
    //       depth should yield lower posterior (rate is halved). All current tests use uniform totals.
    // TODO: test that n_guides_detected reflects the number of guides above min_confidence per cell.

    #[test]
    fn multi_infected_flagged() {
        // Cell 3 has signal in both guides. Uniform depth decouples offset from guide counts.
        let input = make_input_with_totals(8, 2, vec![
            (0,0,1),(0,1,1), (1,0,1),(1,1,1), (2,0,1),
            (3,0,20),(3,1,22),
            (4,0,21), (5,0,20),
            (6,1,23), (7,1,21),
        ], vec![50; 8]);
        let model = PoissonModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(3), "C4 should be multi-infected");
    }

    // ---- posterior_signal (Poisson Bayes) ----

    #[test]
    fn poisson_posterior_equals_pi_when_components_identical() {
        // β1 = 0 ⇒ μ_signal = μ_bg ⇒ likelihoods identical ⇒ posterior = π.
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = FitParams { pi, beta0: -1.0, beta1: 0.0 };
            let post = posterior_signal(3.0, 2.0, p);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn poisson_posterior_monotonic_in_y_when_beta1_positive() {
        let p = FitParams { pi: 0.3, beta0: -2.0, beta1: 2.5 };
        let log_d = (50.0_f64 + 1.0).ln();
        let mut prev = -1.0;
        for y in [0.0, 1.0, 3.0, 8.0, 20.0, 50.0] {
            let post = posterior_signal(y, log_d, p);
            assert!(post > prev, "non-monotonic at y={y}: prev={prev}, post={post}");
            prev = post;
        }
    }

    #[test]
    fn poisson_posterior_saturates() {
        // log_d=0 → μ_bg=e⁰=1, μ_sig=e³≈20. y=50 is overwhelmingly signal.
        let p = FitParams { pi: 0.2, beta0: 0.0, beta1: 3.0 };
        assert!(posterior_signal(50.0, 0.0, p) > 0.99);
        assert!(posterior_signal(0.0, 0.0, p) < 0.2);
    }

    // ---- fit_mixture: synthetic recovery ----

    #[test]
    fn fit_mixture_recovers_bimodal_poisson() {
        // 12 background (counts 0–2) + 8 signal (counts 18–25), uniform depth 50.
        let log_d = (50.0_f64 + 1.0).ln();
        let mut data: Vec<(f64, f64)> = (0..12).map(|i| ((i % 3) as f64, log_d)).collect();
        data.extend((0..8).map(|i| (18.0 + i as f64, log_d)));

        let fitted = fit_mixture(&data, 200, 25, 1e-8);
        let mu_bg = (fitted.beta0 + log_d).exp();
        let mu_sig = (fitted.beta0 + fitted.beta1 + log_d).exp();

        assert!(mu_sig > mu_bg, "μ_sig={mu_sig} ≤ μ_bg={mu_bg}");
        assert!(mu_bg < 5.0, "μ_bg should track low cluster, got {mu_bg}");
        assert!(mu_sig > 15.0, "μ_sig should track high cluster, got {mu_sig}");
        // 8 of 20 = 0.4 signal weight.
        assert!((0.2..=0.6).contains(&fitted.pi), "pi: {}", fitted.pi);
    }
}

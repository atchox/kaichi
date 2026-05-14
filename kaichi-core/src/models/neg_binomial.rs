use super::AssignmentModel;
use super::em::{clamp_probability, log_nb_pmf, logsumexp2, run_em, solve_2x2_sym};
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};
use statrs::function::gamma::digamma;

/// Per-guide Negative Binomial mixture with depth covariate.
///
/// log(μ_i) = β0 + β1·z_i + log(total_counts_i + 1).
/// Overdispersion θ is shared across both components.
/// M-step uses alternating Newton-Raphson: 2D update for (β0, β1) then 1D for log_θ.
pub struct NegBinomialModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub inner_max_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
    pub theta_init: f32,
    pub theta_min: f32,
    pub theta_max: f32,
}

impl Default for NegBinomialModel {
    fn default() -> Self {
        Self {
            min_confidence: 0.8,
            max_em_iters: 100,
            inner_max_iters: 25,
            tol: 1e-6,
            min_nonzero: 2,
            min_max_count: 2,
            theta_init: 5.0,
            theta_min: 0.01,
            theta_max: 1e4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FitParams {
    pi: f64,
    beta0: f64,
    beta1: f64,
    log_theta: f64,
}

impl AssignmentModel for NegBinomialModel {
    fn name(&self) -> &'static str {
        "neg_binomial"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();
        let total_counts = &input.covariates.total_counts;

        let log_depths: Vec<f64> = (0..n_cells)
            .map(|i| (total_counts.value(i) as f64 + 1.0).ln())
            .collect();

        let theta_log_min = (self.theta_min as f64).ln();
        let theta_log_max = (self.theta_max as f64).ln();

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
                self.fit_guide(&data, theta_log_min, theta_log_max)
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
            "theta_init": self.theta_init,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
        })
    }
}

impl NegBinomialModel {
    fn fit_guide(&self, data: &[(f64, f64)], theta_log_min: f64, theta_log_max: f64) -> Option<FitParams> {
        let n_nonzero = data.len() as u32;
        let max_count = data.iter().map(|(y, _)| *y as u32).max().unwrap_or(0);
        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }
        Some(fit_mixture(
            data,
            self.max_em_iters,
            self.inner_max_iters,
            self.tol as f64,
            (self.theta_init as f64).ln(),
            theta_log_min,
            theta_log_max,
        ))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

fn fit_mixture(
    data: &[(f64, f64)],
    max_em_iters: u32,
    inner_max_iters: u32,
    tol: f64,
    log_theta_init: f64,
    log_theta_min: f64,
    log_theta_max: f64,
) -> FitParams {
    let init = initialize_params(data, log_theta_init);
    let mut responsibilities = vec![0.0f64; data.len()];

    run_em(
        init,
        |params| {
            let theta = params.log_theta.exp();
            let mut log_lik = 0.0;
            for (idx, &(y, log_d)) in data.iter().enumerate() {
                let mu0 = (params.beta0 + log_d).exp();
                let mu1 = (params.beta0 + params.beta1 + log_d).exp();
                let log_bg = (1.0 - params.pi).ln() + log_nb_pmf(y, mu0, theta);
                let log_sig = params.pi.ln() + log_nb_pmf(y, mu1, theta);
                let denom = logsumexp2(log_bg, log_sig);
                responsibilities[idx] = (log_sig - denom).exp();
                log_lik += denom;
            }
            let new_params = m_step(
                params, data, &responsibilities, inner_max_iters,
                log_theta_min, log_theta_max,
            );
            (new_params, log_lik)
        },
        max_em_iters,
        tol,
    )
}

fn initialize_params(data: &[(f64, f64)], log_theta_init: f64) -> FitParams {
    let max_y: f64 = data.iter().map(|(y, _)| *y).fold(f64::NEG_INFINITY, f64::max);
    let mean_depth: f64 = data.iter().map(|(_, d)| d.exp()).sum::<f64>() / data.len() as f64;
    let mut sorted_y: Vec<f64> = data.iter().map(|(y, _)| *y).collect();
    sorted_y.sort_by(f64::total_cmp);
    // Use median as robust background-mean estimate (NB has fat tails; mean is pulled up by signal).
    let median_y = sorted_y[sorted_y.len() / 2].max(1.0);
    let beta0 = (median_y / mean_depth.max(1e-6)).ln().clamp(-10.0, 10.0);
    let beta1 = (max_y / median_y).ln().max(0.1);
    FitParams { pi: 0.1, beta0, beta1, log_theta: log_theta_init }
}

fn m_step(
    params: FitParams,
    data: &[(f64, f64)],
    resp: &[f64],
    inner_max_iters: u32,
    log_theta_min: f64,
    log_theta_max: f64,
) -> FitParams {
    let pi = clamp_probability(resp.iter().sum::<f64>() / data.len() as f64);
    let mut beta0 = params.beta0;
    let mut beta1 = params.beta1;
    let mut log_theta = params.log_theta;

    for _ in 0..inner_max_iters {
        let theta = log_theta.exp();

        // Newton step for (β0, β1) using NB Fisher information.
        let mut g = [0.0f64; 2];
        let mut fi = [0.0f64; 3];
        for (&(y, log_d), &r) in data.iter().zip(resp) {
            let mu0 = (beta0 + log_d).exp();
            let mu1 = (beta0 + beta1 + log_d).exp();
            // Score function ∂log_nb/∂μ = y/μ - (y+θ)/(μ+θ). Multiply by μ for β0/β1.
            let s0 = y - mu0 * (y + theta) / (mu0 + theta);
            let s1 = y - mu1 * (y + theta) / (mu1 + theta);
            g[0] += (1.0 - r) * s0 + r * s1;
            g[1] += r * s1;
            // Fisher information: μ·θ/(μ+θ) per GLM IRLS weight.
            let w0 = mu0 * theta / (mu0 + theta);
            let w1 = mu1 * theta / (mu1 + theta);
            fi[0] += (1.0 - r) * w0 + r * w1;
            fi[1] += r * w1;
            fi[2] += r * w1;
        }
        if let Some(delta) = solve_2x2_sym(fi, g) {
            beta0 += delta[0].clamp(-3.0, 3.0);
            beta1 += delta[1].clamp(-3.0, 3.0);
        }

        // 1D Newton step for log_θ.
        let theta = log_theta.exp();
        let mut grad_phi = 0.0;
        let mut hess_phi = 0.0;
        for (&(y, log_d), &r) in data.iter().zip(resp) {
            let mu0 = (beta0 + log_d).exp();
            let mu1 = (beta0 + beta1 + log_d).exp();
            for (mu, w) in [(mu0, 1.0 - r), (mu1, r)] {
                // ∂log_nb/∂θ
                let g_theta = digamma(y + theta) - digamma(theta)
                    + (theta / (theta + mu)).ln() + 1.0 - (theta + y) / (theta + mu);
                // ∂²log_nb/∂θ²
                let h_theta = trigamma(y + theta) - trigamma(theta)
                    + mu / (theta * (theta + mu))
                    + (y - mu) / (theta + mu).powi(2);
                // Chain rule for φ = log_θ: ∂f/∂φ = θ·∂f/∂θ, ∂²f/∂φ² = θ·∂f/∂θ + θ²·∂²f/∂θ²
                grad_phi += w * theta * g_theta;
                hess_phi += w * (theta * g_theta + theta * theta * h_theta);
            }
        }
        if hess_phi.abs() > 1e-14 {
            let step = (-grad_phi / hess_phi).clamp(-1.0, 1.0);
            log_theta = (log_theta + step).clamp(log_theta_min, log_theta_max);
        }
    }

    if beta1 < 0.0 { beta1 = -beta1; }

    FitParams { pi, beta0, beta1, log_theta }
}

fn posterior_signal(y: f64, log_d: f64, params: FitParams) -> f64 {
    let theta = params.log_theta.exp();
    let mu0 = (params.beta0 + log_d).exp();
    let mu1 = (params.beta0 + params.beta1 + log_d).exp();
    let log_bg = (1.0 - params.pi).ln() + log_nb_pmf(y, mu0, theta);
    let log_sig = params.pi.ln() + log_nb_pmf(y, mu1, theta);
    let denom = logsumexp2(log_bg, log_sig);
    (log_sig - denom).exp()
}

/// ψ'(x) = trigamma function via recurrence + asymptotic expansion.
fn trigamma(x: f64) -> f64 {
    let mut x = x;
    let mut sum = 0.0;
    while x < 6.0 {
        sum += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    sum + inv + 0.5 * inv2 + inv * inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 / 42.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, Covariates, GuideMetadata};
    use arrow::array::{BooleanArray, Float32Array, StringBuilder};

    fn make_input(n_cells: usize, n_guides: usize, triples: Vec<(usize, usize, u32)>) -> LoadedInput {
        let mut sorted = triples;
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = sorted.len();
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut last = 0usize;
        let mut cell_totals = vec![0u32; n_cells];
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r { row_offsets[last + 1] = idx; last += 1; }
            col_indices.push(c);
            values.push(v);
            cell_totals[r] += v;
        }
        for i in (last + 1)..=n_cells { row_offsets[i] = nnz; }
        let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        let mut bc = StringBuilder::new();
        for i in 0..n_cells { bc.append_value(format!("C{i}")); }
        let mut gd = StringBuilder::new();
        for i in 0..n_guides { gd.append_value(format!("g{i}")); }
        LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(cell_totals.iter().map(|&t| t as f32).collect::<Vec<_>>()),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn make_input_with_totals(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        totals: Vec<u32>,
    ) -> LoadedInput {
        let mut sorted = triples;
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = sorted.len();
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut last = 0usize;
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r { row_offsets[last + 1] = idx; last += 1; }
            col_indices.push(c);
            values.push(v);
        }
        for i in (last + 1)..=n_cells { row_offsets[i] = nnz; }
        let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        let mut bc = StringBuilder::new();
        for i in 0..n_cells { bc.append_value(format!("C{i}")); }
        let mut gd = StringBuilder::new();
        for i in 0..n_guides { gd.append_value(format!("g{i}")); }
        LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(totals.iter().map(|&t| t as f32).collect::<Vec<_>>()),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    #[test]
    fn assigns_high_signal_cells() {
        // 5 background (y=1) + 3 signal (y=20-25), totals=50.
        // More background than signal ensures median stays in bg → good beta1 init.
        let input = make_input_with_totals(
            8, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,1),(4,0,1),(5,0,20),(6,0,22),(7,0,25)],
            vec![50; 8],
        );
        let model = NegBinomialModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        for i in 0..5 { assert!(is_u.value(i), "cell {i} should be unassigned"); }
        assert!(!is_u.value(5));
        assert!(!is_u.value(6));
        assert!(!is_u.value(7));
    }

    #[test]
    fn skips_guides_with_too_little_signal() {
        let input = make_input(3, 1, vec![(1,0,1),(2,0,1)]);
        let result = NegBinomialModel::default().assign(&input).unwrap();
        for i in 0..3 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = NegBinomialModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        let input = make_input(3, 1, vec![(1,0,1),(2,0,1)]);
        let result = NegBinomialModel::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
            assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(i));
        }
    }

    // TODO: test that depth covariate actually affects assignment — same raw count at 2×
    //       depth should yield lower posterior. All current tests use uniform totals.
    // TODO: test that n_guides_detected reflects the number of guides above min_confidence per cell.

    #[test]
    fn trigamma_positive() {
        // ψ'(1) = π²/6 ≈ 1.6449. ψ'(2) = π²/6 - 1 ≈ 0.6449.
        let t1 = trigamma(1.0);
        let t2 = trigamma(2.0);
        assert!((t1 - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-4, "trigamma(1) ≈ π²/6");
        assert!((t2 - (std::f64::consts::PI.powi(2) / 6.0 - 1.0)).abs() < 1e-4, "trigamma(2)");
    }
}

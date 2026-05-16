use super::AssignmentModel;
use super::em::{clamp_probability, log_nb_pmf, logsumexp2, run_em};
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

#[derive(Clone, Debug)]
struct FitParams {
    pi: f64,
    beta0: f64,
    beta1: f64,
    log_theta: f64,
    /// Per-batch log-rate offset; `gamma[0]` is anchored at 0 for identifiability.
    gamma: Vec<f64>,
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
        let batch = &input.covariates.batch;
        let n_batches = batch.n_batches();

        let log_depths: Vec<f64> = (0..n_cells)
            .map(|i| (total_counts.value(i) as f64 + 1.0).ln())
            .collect();

        let theta_log_min = (self.theta_min as f64).ln();
        let theta_log_max = (self.theta_max as f64).ln();

        let guide_fits: Vec<Option<FitParams>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let data: Vec<(f64, f64, u16)> = col
                    .row_indices()
                    .iter()
                    .zip(col.values())
                    .filter(|(_, &v)| v > 0)
                    .map(|(&i, &v)| (v as f64, log_depths[i], batch.codes[i]))
                    .collect();
                self.fit_guide(&data, n_batches, theta_log_min, theta_log_max)
            })
            .collect();

        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let log_d = log_depths[cell];
            let b = batch.codes[cell] as usize;
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                let params = match &guide_fits[guide_idx] {
                    Some(p) => p,
                    None => continue,
                };
                let post = posterior_signal(count as f64, log_d, params, b) as f32;
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
    fn fit_guide(&self, data: &[(f64, f64, u16)], n_batches: usize, theta_log_min: f64, theta_log_max: f64) -> Option<FitParams> {
        let n_nonzero = data.len() as u32;
        let max_count = data.iter().map(|(y, _, _)| *y as u32).max().unwrap_or(0);
        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }
        Some(fit_mixture(
            data,
            n_batches,
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
    data: &[(f64, f64, u16)],
    n_batches: usize,
    max_em_iters: u32,
    inner_max_iters: u32,
    tol: f64,
    log_theta_init: f64,
    log_theta_min: f64,
    log_theta_max: f64,
) -> FitParams {
    let init = initialize_params(data, n_batches, log_theta_init);
    let mut responsibilities = vec![0.0f64; data.len()];

    run_em(
        init,
        |params| {
            let theta = params.log_theta.exp();
            let mut log_lik = 0.0;
            for (idx, &(y, log_d, b)) in data.iter().enumerate() {
                let gb = params.gamma[b as usize];
                let mu0 = (params.beta0 + gb + log_d).exp();
                let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
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

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize, log_theta_init: f64) -> FitParams {
    let max_y: f64 = data.iter().map(|(y, _, _)| *y).fold(f64::NEG_INFINITY, f64::max);
    let mean_depth: f64 = data.iter().map(|(_, d, _)| d.exp()).sum::<f64>() / data.len() as f64;

    // Per-batch median y → per-batch baseline init. Without this, a global median over
    // heterogeneous batches puts the EM in the wrong basin.
    let mut per_batch_y: Vec<Vec<f64>> = vec![Vec::new(); n_batches];
    for &(y, _, b) in data {
        if (b as usize) < n_batches {
            per_batch_y[b as usize].push(y);
        }
    }
    let per_batch_median: Vec<f64> = per_batch_y
        .iter_mut()
        .map(|ys| {
            if ys.is_empty() {
                1.0
            } else {
                ys.sort_by(f64::total_cmp);
                ys[ys.len() / 2].max(1.0)
            }
        })
        .collect();

    let m0 = per_batch_median[0];
    let beta0 = (m0 / mean_depth.max(1e-6)).ln().clamp(-10.0, 10.0);
    let mut gamma = vec![0.0; n_batches];
    for b in 1..n_batches {
        gamma[b] = (per_batch_median[b] / m0).ln().clamp(-5.0, 5.0);
    }

    let mut all_y: Vec<f64> = data.iter().map(|(y, _, _)| *y).collect();
    all_y.sort_by(f64::total_cmp);
    let global_median = all_y[all_y.len() / 2].max(1.0);
    let beta1 = (max_y / global_median).ln().max(0.1);

    FitParams { pi: 0.1, beta0, beta1, log_theta: log_theta_init, gamma }
}

fn m_step(
    params: FitParams,
    data: &[(f64, f64, u16)],
    resp: &[f64],
    inner_max_iters: u32,
    log_theta_min: f64,
    log_theta_max: f64,
) -> FitParams {
    let pi = clamp_probability(resp.iter().sum::<f64>() / data.len() as f64);
    let n_batches = params.gamma.len();

    // Same arrow-Hessian Schur strategy as Poisson, but with NB's score (θ(y−μ)/(μ+θ))
    // and Fisher info (μθ/(μ+θ)). η_b = β0 + γ_b carries the per-batch baseline.
    let mut eta: Vec<f64> = (0..n_batches)
        .map(|b| params.beta0 + params.gamma[b])
        .collect();
    let mut beta1 = params.beta1;
    let mut log_theta = params.log_theta;

    for _ in 0..inner_max_iters {
        let theta = log_theta.exp();

        // (η_0, ..., η_{B−1}, β1) Schur step with NB weights, holding θ fixed.
        let mut g_eta = vec![0.0f64; n_batches];
        let mut d = vec![0.0f64; n_batches];
        let mut c = vec![0.0f64; n_batches];
        let mut g_beta1 = 0.0f64;
        let mut s = 0.0f64;

        for (&(y, log_d, bi), &r) in data.iter().zip(resp) {
            let b = bi as usize;
            let mu0 = (eta[b] + log_d).exp();
            let mu1 = (eta[b] + beta1 + log_d).exp();
            let s0 = theta * (y - mu0) / (mu0 + theta);
            let s1 = theta * (y - mu1) / (mu1 + theta);
            g_eta[b] += (1.0 - r) * s0 + r * s1;
            g_beta1 += r * s1;
            let w0 = mu0 * theta / (mu0 + theta);
            let w1 = mu1 * theta / (mu1 + theta);
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

        for b in 0..n_batches {
            let d_eta = if d[b] > 1e-14 {
                ((g_eta[b] - c[b] * d_beta1) / d[b]).clamp(-3.0, 3.0)
            } else { 0.0 };
            eta[b] += d_eta;
        }
        beta1 += d_beta1;

        // 1D Newton step for log_θ (holding η and β1 fixed). NB-specific digamma/trigamma math.
        let theta = log_theta.exp();
        let mut grad_phi = 0.0;
        let mut hess_phi = 0.0;
        for (&(y, log_d, bi), &r) in data.iter().zip(resp) {
            let b = bi as usize;
            let mu0 = (eta[b] + log_d).exp();
            let mu1 = (eta[b] + beta1 + log_d).exp();
            for (mu, w) in [(mu0, 1.0 - r), (mu1, r)] {
                let g_theta = digamma(y + theta) - digamma(theta)
                    + (theta / (theta + mu)).ln() + 1.0 - (theta + y) / (theta + mu);
                let h_theta = trigamma(y + theta) - trigamma(theta)
                    + mu / (theta * (theta + mu))
                    + (y - mu) / (theta + mu).powi(2);
                grad_phi += w * theta * g_theta;
                hess_phi += w * (theta * g_theta + theta * theta * h_theta);
            }
        }
        if hess_phi.abs() > 1e-14 {
            let step = (-grad_phi / hess_phi).clamp(-1.0, 1.0);
            log_theta = (log_theta + step).clamp(log_theta_min, log_theta_max);
        }
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

    FitParams { pi, beta0, beta1, log_theta, gamma }
}

fn posterior_signal(y: f64, log_d: f64, params: &FitParams, batch_idx: usize) -> f64 {
    let theta = params.log_theta.exp();
    let gb = params.gamma[batch_idx];
    let mu0 = (params.beta0 + gb + log_d).exp();
    let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
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

    #[test]
    fn n_guides_detected_counts_guides_above_threshold() {
        // Cells 0..3 are test subjects (0/1/2/3 guides above threshold). Cells 4..7
        // are background ballast — NB's overdispersion absorbs more of the signal/bg
        // gap than Poisson's does, so we need a clear majority of small-count cells
        // for the EM to place its β0 in the background regime.
        let mut triples = vec![
            (0, 0, 1u32), (0, 1, 1), (0, 2, 1),
            (1, 0, 20),
            (2, 0, 21), (2, 1, 22),
            (3, 0, 20), (3, 1, 21), (3, 2, 22),
        ];
        for c in 4..8 {
            for g in 0..3 { triples.push((c, g, 1)); }
        }
        let input = make_input_with_totals(8, 3, triples, vec![50; 8]);
        let model = NegBinomialModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
        assert_eq!(n_det.value(0), 0);
        assert_eq!(n_det.value(1), 1);
        assert_eq!(n_det.value(2), 2);
        assert_eq!(n_det.value(3), 3);
    }

    #[test]
    fn trigamma_positive() {
        // ψ'(1) = π²/6 ≈ 1.6449. ψ'(2) = π²/6 - 1 ≈ 0.6449.
        let t1 = trigamma(1.0);
        let t2 = trigamma(2.0);
        assert!((t1 - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-4, "trigamma(1) ≈ π²/6");
        assert!((t2 - (std::f64::consts::PI.powi(2) / 6.0 - 1.0)).abs() < 1e-4, "trigamma(2)");
    }

    #[test]
    fn trigamma_recurrence_holds() {
        // ψ'(x+1) = ψ'(x) - 1/x². Verified across the recurrence regime (<6) and the
        // asymptotic regime (≥6) — independent of which branch the implementation hits.
        // The asymptotic expansion has ~1e-7 truncation error at the recurrence boundary;
        // away from x=6 the error drops to machine precision. 1e-6 is loose enough for the
        // boundary and still tight enough to catch any real implementation bug.
        for &x in &[0.5, 1.0, 1.5, 2.7, 3.3, 4.9, 5.9, 6.0, 6.5, 12.3, 100.0] {
            let lhs = trigamma(x + 1.0);
            let rhs = trigamma(x) - 1.0 / (x * x);
            assert!((lhs - rhs).abs() < 1e-6, "x={x}: lhs={lhs}, rhs={rhs}, diff={}", (lhs - rhs).abs());
        }
    }

    // ---- posterior_signal (NB Bayes) ----

    fn fp(pi: f64, beta0: f64, beta1: f64, log_theta: f64) -> FitParams {
        FitParams { pi, beta0, beta1, log_theta, gamma: vec![0.0] }
    }

    #[test]
    fn nb_posterior_equals_pi_when_components_identical() {
        // β1 = 0 ⇒ μ_signal = μ_bg ⇒ likelihoods identical ⇒ posterior = π.
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = fp(pi, -1.0, 0.0, 1.0);
            let post = posterior_signal(3.0, 2.0, &p, 0);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn nb_posterior_monotonic_in_y_when_beta1_positive() {
        // With β1 > 0 (signal mean > background mean), bigger y favors signal.
        let p = fp(0.3, -1.5, 2.0, 1.0);
        let log_d = (50.0_f64 + 1.0).ln();
        let mut prev = -1.0;
        for y in [0.0, 1.0, 3.0, 8.0, 20.0, 50.0] {
            let post = posterior_signal(y, log_d, &p, 0);
            assert!(post > prev, "non-monotonic at y={y}: prev={prev}, post={post}");
            prev = post;
        }
    }

    #[test]
    fn nb_posterior_saturates_at_very_large_y() {
        let p = fp(0.2, 0.0, 3.0, 5.0);
        assert!(posterior_signal(50.0, 0.0, &p, 0) > 0.99);
        assert!(posterior_signal(0.0, 0.0, &p, 0) < 0.2);
    }

    #[test]
    fn nb_posterior_drops_when_depth_doubles() {
        // Same identity as Poisson: doubling depth (log_d → log_d + ln 2) doubles both
        // μ_bg and μ_sig, leaving β1 unchanged but growing μ_sig − μ_bg. A fixed y looks
        // less surprising under the larger background, so the signal posterior falls.
        let p = fp(0.3, -2.0, 2.0, 1.0);
        let log_d_lo = (10.0_f64 + 1.0).ln();
        let log_d_hi = (21.0_f64 + 1.0).ln();
        let y = 4.0;
        let post_lo = posterior_signal(y, log_d_lo, &p, 0);
        let post_hi = posterior_signal(y, log_d_hi, &p, 0);
        assert!(
            post_hi < post_lo,
            "doubling depth should lower posterior at fixed y: lo={post_lo}, hi={post_hi}"
        );
    }

    #[test]
    fn nb_posterior_uses_batch_gamma() {
        // γ_1 > 0 raises batch 1's baseline → the same y at log_d=0 looks more like
        // background in batch 1 than in batch 0.
        let p = FitParams { pi: 0.3, beta0: -2.0, beta1: 2.0, log_theta: 1.0, gamma: vec![0.0, 1.5] };
        let b0 = posterior_signal(5.0, 0.0, &p, 0);
        let b1 = posterior_signal(5.0, 0.0, &p, 1);
        assert!(b0 > b1, "γ_1 > 0 should suppress signal in batch 1: b0={b0}, b1={b1}");
    }

    // ---- fit_mixture: synthetic recovery ----

    #[test]
    fn fit_mixture_recovers_bimodal_nb() {
        // 15 background counts 0–2, 10 signal counts 18–27. All cells at depth 50, single batch.
        let log_d = (50.0_f64 + 1.0).ln();
        let mut data: Vec<(f64, f64, u16)> = (0..15).map(|i| ((i % 3) as f64, log_d, 0u16)).collect();
        data.extend((0..10).map(|i| (18.0 + i as f64, log_d, 0u16)));

        let fitted = fit_mixture(
            &data, 1, 200, 25, 1e-8,
            5.0_f64.ln(), 0.01_f64.ln(), 1e4_f64.ln(),
        );

        let mu_bg = (fitted.beta0 + log_d).exp();
        let mu_sig = (fitted.beta0 + fitted.beta1 + log_d).exp();

        assert!(mu_sig > mu_bg, "μ_sig={mu_sig} should exceed μ_bg={mu_bg}");
        assert!(mu_bg < 5.0, "μ_bg should track the low cluster, got {mu_bg}");
        assert!(mu_sig > 15.0, "μ_sig should track the high cluster, got {mu_sig}");
        assert!((0.2..=0.6).contains(&fitted.pi), "pi drifted: {}", fitted.pi);
        assert_eq!(fitted.gamma, vec![0.0]);
    }

    #[test]
    fn fit_mixture_respects_theta_clamp() {
        let log_d = (100.0_f64 + 1.0).ln();
        let data: Vec<(f64, f64, u16)> = (0..15)
            .map(|i| if i < 10 { 1.0 } else { 20.0 })
            .map(|y| (y, log_d, 0u16))
            .collect();
        let cap_log = 3.0;
        let fitted = fit_mixture(&data, 1, 200, 25, 1e-8, 1.0, -4.0, cap_log);
        assert!(
            fitted.log_theta <= cap_log + 1e-9,
            "log_theta {} exceeded cap {cap_log}", fitted.log_theta
        );
    }

    #[test]
    fn fit_mixture_recovers_batch_offsets() {
        // Two batches with same signal/background shape but 4× baselines.
        // γ_1 should land near ln(4) ≈ 1.39.
        let log_d = (50.0_f64 + 1.0).ln();
        let mut data: Vec<(f64, f64, u16)> = Vec::new();
        for i in 0..10 { data.push(((i % 2 + 1) as f64, log_d, 0u16)); }
        for i in 0..6  { data.push((18.0 + i as f64, log_d, 0u16)); }
        for i in 0..10 { data.push(((i % 2 + 4) as f64, log_d, 1u16)); }
        for i in 0..6  { data.push((75.0 + i as f64, log_d, 1u16)); }

        let fitted = fit_mixture(
            &data, 2, 200, 50, 1e-8,
            5.0_f64.ln(), 0.01_f64.ln(), 1e4_f64.ln(),
        );
        assert_eq!(fitted.gamma[0], 0.0);
        // Truth γ_1 = ln(4) ≈ 1.386; the EM lands near 1.307 here. Same tolerance as Poisson.
        assert!(
            (fitted.gamma[1] - 4.0_f64.ln()).abs() < 0.15,
            "γ_1 expected ≈ ln(4)={:.3}, got {:.3}", 4.0_f64.ln(), fitted.gamma[1]
        );
    }
}

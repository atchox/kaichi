use super::AssignmentModel;
use super::em::{clamp_probability, log_nb_pmf, logsumexp2};
use super::em_count_mixture::{decide_threshold, score_em_count_mixture, EmConfig, EmCountMixture, Gating, GuideData};
use super::TwoStage;
use crate::data::{AssignmentResult, LoadedInput};
use crate::score::{ModelKind, ScoreMatrix};

use anyhow::Result;
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
    pub n_restarts: u32,
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
            n_restarts: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FitParams {
    pub pi: f64,
    pub beta0: f64,
    pub beta1: f64,
    pub log_theta: f64,
    /// Per-batch log-rate offset; `gamma[0]` is anchored at 0 for identifiability.
    pub gamma: Vec<f64>,
}

// ---------------------------------------------------------------------------
// AssignmentModel
// ---------------------------------------------------------------------------

impl AssignmentModel for NegBinomialModel {
    fn name(&self) -> &'static str {
        "neg_binomial"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let scores = self.score(input)?;
        self.decide(&scores, self.min_confidence)
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
            "n_restarts": self.n_restarts,
        })
    }
}

// ---------------------------------------------------------------------------
// TwoStage
// ---------------------------------------------------------------------------

impl TwoStage for NegBinomialModel {
    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix> {
        score_em_count_mixture(self, ModelKind::NegBinomial, self.params_json(), input)
    }

    fn decide(&self, scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult> {
        decide_threshold(scores, min_confidence)
    }
}

// ---------------------------------------------------------------------------
// EmCountMixture
// ---------------------------------------------------------------------------

impl EmCountMixture for NegBinomialModel {
    type FitParams = FitParams;

    fn gating(&self) -> Gating {
        Gating { min_nonzero: self.min_nonzero, min_max_count: self.min_max_count }
    }

    fn em_config(&self) -> EmConfig {
        EmConfig {
            max_iters: self.max_em_iters,
            n_restarts: self.n_restarts,
            tol: self.tol as f64,
        }
    }

    fn init(&self, data: &GuideData, n_batches: usize, pi_k: f64) -> FitParams {
        let mut p = initialize_params(&data.triples, n_batches, (self.theta_init as f64).ln());
        p.pi = pi_k;
        p
    }

    fn em_cycle(
        &self,
        params: FitParams,
        data: &GuideData,
        _n_batches: usize,
        responsibilities: &mut Vec<f64>,
    ) -> (FitParams, f64) {
        let log_theta_min = (self.theta_min as f64).ln();
        let log_theta_max = (self.theta_max as f64).ln();
        let theta = params.log_theta.exp();
        let mut ll = 0.0;
        for (idx, &(y, log_d, b)) in data.triples.iter().enumerate() {
            let gb = params.gamma[b as usize];
            let mu0 = (params.beta0 + gb + log_d).exp();
            let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
            let log_bg = (1.0 - params.pi).ln() + log_nb_pmf(y, mu0, theta);
            let log_sig = params.pi.ln() + log_nb_pmf(y, mu1, theta);
            let denom = logsumexp2(log_bg, log_sig);
            responsibilities[idx] = (log_sig - denom).exp();
            ll += denom;
        }
        let new_params = m_step(
            params,
            &data.triples,
            responsibilities,
            self.inner_max_iters,
            log_theta_min,
            log_theta_max,
        );
        (new_params, ll)
    }

    fn posterior(&self, count: u32, covariate: f64, params: &FitParams, batch: u16) -> f32 {
        posterior_signal(count as f64, covariate, params, batch as usize) as f32
    }
}

// ---------------------------------------------------------------------------
// EM math (unchanged)
// ---------------------------------------------------------------------------

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize, log_theta_init: f64) -> FitParams {
    let max_y: f64 = data.iter().map(|(y, _, _)| *y).fold(f64::NEG_INFINITY, f64::max);
    let mean_depth: f64 = data.iter().map(|(_, d, _)| d.exp()).sum::<f64>() / data.len() as f64;

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

    let mut eta: Vec<f64> = (0..n_batches)
        .map(|b| params.beta0 + params.gamma[b])
        .collect();
    let mut beta1 = params.beta1;
    let mut log_theta = params.log_theta;

    for _ in 0..inner_max_iters {
        let theta = log_theta.exp();

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
        } else {
            0.0
        };

        for b in 0..n_batches {
            let d_eta = if d[b] > 1e-14 {
                ((g_eta[b] - c[b] * d_beta1) / d[b]).clamp(-3.0, 3.0)
            } else {
                0.0
            };
            eta[b] += d_eta;
        }
        beta1 += d_beta1;

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
        let t1 = trigamma(1.0);
        let t2 = trigamma(2.0);
        assert!((t1 - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-4, "trigamma(1) ≈ π²/6");
        assert!((t2 - (std::f64::consts::PI.powi(2) / 6.0 - 1.0)).abs() < 1e-4, "trigamma(2)");
    }

    #[test]
    fn trigamma_recurrence_holds() {
        for &x in &[0.5, 1.0, 1.5, 2.7, 3.3, 4.9, 5.9, 6.0, 6.5, 12.3, 100.0] {
            let lhs = trigamma(x + 1.0);
            let rhs = trigamma(x) - 1.0 / (x * x);
            assert!((lhs - rhs).abs() < 1e-6, "x={x}: lhs={lhs}, rhs={rhs}, diff={}", (lhs - rhs).abs());
        }
    }

    // ---- posterior_signal ----

    fn fp(pi: f64, beta0: f64, beta1: f64, log_theta: f64) -> FitParams {
        FitParams { pi, beta0, beta1, log_theta, gamma: vec![0.0] }
    }

    #[test]
    fn nb_posterior_equals_pi_when_components_identical() {
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = fp(pi, -1.0, 0.0, 1.0);
            let post = posterior_signal(3.0, 2.0, &p, 0);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn nb_posterior_monotonic_in_y_when_beta1_positive() {
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
        let p = FitParams { pi: 0.3, beta0: -2.0, beta1: 2.0, log_theta: 1.0, gamma: vec![0.0, 1.5] };
        let b0 = posterior_signal(5.0, 0.0, &p, 0);
        let b1 = posterior_signal(5.0, 0.0, &p, 1);
        assert!(b0 > b1, "γ_1 > 0 should suppress signal in batch 1: b0={b0}, b1={b1}");
    }

    #[test]
    fn em_recovers_batch_offsets() {
        use super::super::em::run_em;
        use super::super::em_count_mixture::GuideData;
        let log_d = (50.0_f64 + 1.0).ln();
        let model = NegBinomialModel::default();
        // Two batches, batch 1 has ~doubled rates ⇒ true γ_1 ≈ ln(2).
        let mut data: Vec<(f64, f64, u16)> = Vec::new();
        for i in 0..12 { data.push(((i % 3) as f64, log_d, 0u16)); }
        for i in 0..8  { data.push((18.0 + i as f64, log_d, 0u16)); }
        for i in 0..12 { data.push(((i % 3) as f64 + 1.0, log_d, 1u16)); }
        for i in 0..8  { data.push((36.0 + i as f64, log_d, 1u16)); }
        let guide_data = GuideData { triples: data.clone(), n_zeros: 0 };

        let mut resp = vec![0.0f64; data.len()];
        let init = model.init(&guide_data, 2, 0.1);
        let (fitted, _) = run_em(
            init,
            |p| model.em_cycle(p, &guide_data, 2, &mut resp),
            300,
            1e-9,
        );

        assert_eq!(fitted.gamma.len(), 2);
        assert_eq!(fitted.gamma[0], 0.0, "γ_0 anchored at 0");
        assert!(
            fitted.gamma[1] > 0.3,
            "γ_1 should be positive when batch 1 has elevated rates; got {}",
            fitted.gamma[1]
        );
    }

    #[test]
    fn em_respects_theta_clamp() {
        use super::super::em::run_em;
        use super::super::em_count_mixture::GuideData;
        let log_d = (50.0_f64 + 1.0).ln();
        // Tight clamp far from any reasonable MLE for clean bimodal data.
        let model = NegBinomialModel {
            theta_init: 0.75,
            theta_min: 0.5,
            theta_max: 1.0,
            ..Default::default()
        };
        let mut data: Vec<(f64, f64, u16)> = (0..12).map(|i| ((i % 3) as f64, log_d, 0u16)).collect();
        data.extend((0..8).map(|i| (18.0 + i as f64, log_d, 0u16)));
        let guide_data = GuideData { triples: data.clone(), n_zeros: 0 };

        let mut resp = vec![0.0f64; data.len()];
        let init = model.init(&guide_data, 1, 0.1);
        let (fitted, _) = run_em(
            init,
            |p| model.em_cycle(p, &guide_data, 1, &mut resp),
            300,
            1e-9,
        );

        let theta = fitted.log_theta.exp();
        assert!(
            theta >= 0.5 - 1e-9 && theta <= 1.0 + 1e-9,
            "θ should stay within [0.5, 1.0]; got {theta}"
        );
    }

    #[test]
    fn score_and_decide_matches_assign() {
        let input = make_input_with_totals(
            8, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,1),(4,0,1),(5,0,20),(6,0,22),(7,0,25)],
            vec![50; 8],
        );
        let model = NegBinomialModel { min_confidence: 0.8, ..Default::default() };
        let direct = model.assign(&input).unwrap();
        let scores = model.score(&input).unwrap();
        let via_decide = model.decide(&scores, 0.8).unwrap();
        let is_u_d = direct.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let is_u_s = via_decide.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..8 {
            assert_eq!(is_u_d.value(i), is_u_s.value(i), "cell {i}");
        }
    }
}

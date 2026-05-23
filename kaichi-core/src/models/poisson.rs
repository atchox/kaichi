use super::AssignmentModel;
use super::em::{clamp_probability, log_poisson_pmf, logsumexp2};
use super::em_count_mixture::{decide_threshold, score_em_count_mixture, EmConfig, EmCountMixture, Gating, GuideData};
use super::TwoStage;
use crate::data::{AssignmentResult, LoadedInput};
use crate::score::{ModelKind, ScoreMatrix};

use anyhow::Result;
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
    pub n_restarts: u32,
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
            n_restarts: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FitParams {
    pub pi: f64,
    pub beta0: f64,
    pub beta1: f64,
    /// Per-batch log-rate offset. `gamma[0]` is anchored at 0 for identifiability.
    pub gamma: Vec<f64>,
}

// ---------------------------------------------------------------------------
// AssignmentModel — delegates through TwoStage
// ---------------------------------------------------------------------------

impl AssignmentModel for PoissonModel {
    fn name(&self) -> &'static str {
        "poisson"
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
            "n_restarts": self.n_restarts,
        })
    }
}

// ---------------------------------------------------------------------------
// TwoStage
// ---------------------------------------------------------------------------

impl TwoStage for PoissonModel {
    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix> {
        score_em_count_mixture(self, ModelKind::Poisson, self.params_json(), input)
    }

    fn decide(&self, scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult> {
        decide_threshold(scores, min_confidence)
    }
}

// ---------------------------------------------------------------------------
// EmCountMixture
// ---------------------------------------------------------------------------

impl EmCountMixture for PoissonModel {
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
        let mut p = initialize_params(&data.triples, n_batches);
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
        let mut ll = 0.0;
        for (idx, &(y, log_d, b)) in data.triples.iter().enumerate() {
            let gb = params.gamma[b as usize];
            let mu0 = (params.beta0 + gb + log_d).exp();
            let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
            let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(y, mu0);
            let log_sig = params.pi.ln() + log_poisson_pmf(y, mu1);
            let denom = logsumexp2(log_bg, log_sig);
            responsibilities[idx] = (log_sig - denom).exp();
            ll += denom;
        }
        let new_params = m_step(params, &data.triples, responsibilities, self.inner_max_iters);
        (new_params, ll)
    }

    fn posterior(&self, count: u32, covariate: f64, params: &FitParams, batch: u16) -> f32 {
        posterior_signal(count as f64, covariate, params, batch as usize) as f32
    }
}

// ---------------------------------------------------------------------------
// EM math (unchanged)
// ---------------------------------------------------------------------------

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize) -> FitParams {
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

        for (&(y, log_d, bi), &r) in data.iter().zip(resp) {
            let b = bi as usize;
            let mu0 = (eta[b] + log_d).exp();
            let mu1 = (eta[b] + beta1 + log_d).exp();
            g_eta[b] += (1.0 - r) * (y - mu0) + r * (y - mu1);
            g_beta1 += r * (y - mu1);
            d[b] += (1.0 - r) * mu0 + r * mu1;
            c[b] += r * mu1;
            s += r * mu1;
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

        let mut max_step = d_beta1.abs();
        for b in 0..n_batches {
            let d_eta = if d[b] > 1e-14 {
                ((g_eta[b] - c[b] * d_beta1) / d[b]).clamp(-3.0, 3.0)
            } else {
                0.0
            };
            eta[b] += d_eta;
            max_step = max_step.max(d_eta.abs());
        }
        beta1 += d_beta1;

        if max_step < 1e-8 {
            break;
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

    FitParams { pi, beta0, beta1, gamma }
}

fn posterior_signal(y: f64, log_d: f64, params: &FitParams, batch_idx: usize) -> f64 {
    let gb = params.gamma[batch_idx];
    let mu0 = (params.beta0 + gb + log_d).exp();
    let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
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
        let model = PoissonModel { min_confidence: 0.8, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
        assert_eq!(n_det.value(0), 0, "background cell has 0 guides detected");
        assert_eq!(n_det.value(1), 1, "single-signal cell has 1 guide detected");
        assert_eq!(n_det.value(2), 2, "double-signal cell has 2 guides detected");
        assert_eq!(n_det.value(3), 3, "triple-signal cell has 3 guides detected");
    }

    #[test]
    fn multi_infected_flagged() {
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

    fn fp(pi: f64, beta0: f64, beta1: f64) -> FitParams {
        FitParams { pi, beta0, beta1, gamma: vec![0.0] }
    }

    #[test]
    fn poisson_posterior_equals_pi_when_components_identical() {
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = fp(pi, -1.0, 0.0);
            let post = posterior_signal(3.0, 2.0, &p, 0);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn poisson_posterior_monotonic_in_y_when_beta1_positive() {
        let p = fp(0.3, -2.0, 2.5);
        let log_d = (50.0_f64 + 1.0).ln();
        let mut prev = -1.0;
        for y in [0.0, 1.0, 3.0, 8.0, 20.0, 50.0] {
            let post = posterior_signal(y, log_d, &p, 0);
            assert!(post > prev, "non-monotonic at y={y}: prev={prev}, post={post}");
            prev = post;
        }
    }

    #[test]
    fn poisson_posterior_saturates() {
        let p = fp(0.2, 0.0, 3.0);
        assert!(posterior_signal(50.0, 0.0, &p, 0) > 0.99);
        assert!(posterior_signal(0.0, 0.0, &p, 0) < 0.2);
    }

    #[test]
    fn poisson_posterior_drops_when_depth_doubles() {
        let p = fp(0.3, -2.0, 2.0);
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
    fn poisson_posterior_uses_batch_gamma() {
        let p = FitParams { pi: 0.3, beta0: -2.0, beta1: 2.0, gamma: vec![0.0, 1.5] };
        let post_b0 = posterior_signal(5.0, 0.0, &p, 0);
        let post_b1 = posterior_signal(5.0, 0.0, &p, 1);
        assert!(post_b0 > post_b1, "γ_1 > 0 should reduce signal posterior in batch 1: b0={post_b0}, b1={post_b1}");
    }

    // ---- fit via EmCountMixture ----

    #[test]
    fn em_mixture_recovers_bimodal_poisson() {
        use super::super::em::run_em;
        use super::super::em_count_mixture::GuideData;
        let log_d = (50.0_f64 + 1.0).ln();
        let model = PoissonModel::default();
        let mut data_vec: Vec<(f64, f64, u16)> =
            (0..12).map(|i| ((i % 3) as f64, log_d, 0u16)).collect();
        data_vec.extend((0..8).map(|i| (18.0 + i as f64, log_d, 0u16)));
        let guide_data = GuideData { triples: data_vec.clone(), n_zeros: 0 };

        let mut responsibilities = vec![0.0f64; data_vec.len()];
        let init = model.init(&guide_data, 1, 0.1);
        let (fitted, _) = run_em(
            init,
            |p| model.em_cycle(p, &guide_data, 1, &mut responsibilities),
            200,
            1e-8,
        );

        let mu_bg = (fitted.beta0 + log_d).exp();
        let mu_sig = (fitted.beta0 + fitted.beta1 + log_d).exp();
        assert!(mu_sig > mu_bg, "μ_sig={mu_sig} ≤ μ_bg={mu_bg}");
        assert!(mu_bg < 5.0, "μ_bg should track low cluster, got {mu_bg}");
        assert!(mu_sig > 15.0, "μ_sig should track high cluster, got {mu_sig}");
    }

    #[test]
    fn score_and_decide_matches_assign() {
        let input = make_input_with_totals(
            6, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,20),(4,0,22),(5,0,25)],
            vec![50,50,50,50,50,50],
        );
        let model = PoissonModel { min_confidence: 0.8, ..Default::default() };
        let direct = model.assign(&input).unwrap();

        let scores = model.score(&input).unwrap();
        let via_decide = model.decide(&scores, 0.8).unwrap();

        // Both should agree on is_unassigned for every cell.
        let is_u_direct = direct.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let is_u_decided = via_decide.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..6 {
            assert_eq!(is_u_direct.value(i), is_u_decided.value(i), "cell {i}");
        }
    }
}

use super::AssignmentModel;
use super::em::{clamp_probability, log_binom_pmf, logsumexp2, sigmoid};
use super::em_count_mixture::{decide_threshold, score_em_count_mixture, EmConfig, EmCountMixture, Gating, GuideData};
use super::TwoStage;
use crate::data::{AssignmentResult, LoadedInput};
use crate::score::{ModelKind, ScoreMatrix};

use anyhow::Result;
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
    pub n_restarts: u32,
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
            n_restarts: 5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FitParams {
    pub pi: f64,
    pub beta0: f64,
    pub beta1: f64,
    /// Per-batch logit offset; `gamma[0]` is anchored at 0 for identifiability.
    pub gamma: Vec<f64>,
}

// ---------------------------------------------------------------------------
// AssignmentModel
// ---------------------------------------------------------------------------

impl AssignmentModel for BinomialModel {
    fn name(&self) -> &'static str {
        "binomial"
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

impl TwoStage for BinomialModel {
    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix> {
        score_em_count_mixture(self, ModelKind::Binomial, self.params_json(), input)
    }

    fn decide(&self, scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult> {
        decide_threshold(scores, min_confidence)
    }
}

// ---------------------------------------------------------------------------
// EmCountMixture
// ---------------------------------------------------------------------------

impl EmCountMixture for BinomialModel {
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

    /// Binomial covariate is the total guide UMI count per cell (trial count),
    /// not log_depth.
    fn prepare_covariates(&self, input: &LoadedInput) -> Vec<f64> {
        let tc = &input.covariates.total_counts;
        (0..input.counts.n_cells)
            .map(|i| tc.value(i) as f64)
            .collect()
    }

    /// Also filter cells where trial count == 0 (no UMIs → Binomial undefined).
    fn guide_data(
        &self,
        col_row_indices: &[usize],
        col_values: &[u32],
        covariates: &[f64],
        batch_codes: &[u16],
        _n_cells: usize,
    ) -> GuideData {
        let triples = col_row_indices
            .iter()
            .zip(col_values)
            .filter(|(&i, &v)| v > 0 && covariates[i] > 0.0)
            .map(|(&i, &v)| (v as f64, covariates[i], batch_codes[i]))
            .collect();
        GuideData { triples, n_zeros: 0 }
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
        // covariate = trial count (n_i), not log_depth
        let mut ll = 0.0;
        for (idx, &(y, n, b)) in data.triples.iter().enumerate() {
            let gb = params.gamma[b as usize];
            let p0 = sigmoid(params.beta0 + gb);
            let p1 = sigmoid(params.beta0 + params.beta1 + gb);
            let log_bg = (1.0 - params.pi).ln() + log_binom_pmf(y, n, p0);
            let log_sig = params.pi.ln() + log_binom_pmf(y, n, p1);
            let denom = logsumexp2(log_bg, log_sig);
            responsibilities[idx] = (log_sig - denom).exp();
            ll += denom;
        }
        let new_params = m_step(params, &data.triples, responsibilities, self.inner_max_iters);
        (new_params, ll)
    }

    fn posterior(&self, count: u32, covariate: f64, params: &FitParams, batch: u16) -> f32 {
        // covariate = trial count
        if covariate <= 0.0 {
            return 0.0;
        }
        posterior_signal(count as f64, covariate, params, batch as usize) as f32
    }
}

// ---------------------------------------------------------------------------
// EM math (unchanged)
// ---------------------------------------------------------------------------

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize) -> FitParams {
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

    // ---- posterior_signal ----

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
        let p = FitParams { pi: 0.3, beta0: -3.0, beta1: 3.0, gamma: vec![0.0, 2.0] };
        let b0 = posterior_signal(5.0, 50.0, &p, 0);
        let b1 = posterior_signal(5.0, 50.0, &p, 1);
        assert!(b0 > b1, "γ_1 > 0 should suppress signal in batch 1: b0={b0}, b1={b1}");
    }

    #[test]
    fn em_recovers_batch_offsets() {
        use super::super::em::run_em;
        use super::super::em_count_mixture::GuideData;
        let model = BinomialModel::default();
        // Trial count = 50; batch 1 has elevated proportions in both components
        // (≈ positive logit offset) ⇒ γ_1 should fit > 0.
        let n = 50.0_f64;
        let mut data: Vec<(f64, f64, u16)> = Vec::new();
        for i in 0..12 { data.push(((i % 3) as f64, n, 0u16)); }            // bg b0, p~0.02
        for i in 0..8  { data.push((18.0 + i as f64, n, 0u16)); }           // sig b0, p~0.43
        for i in 0..12 { data.push((2.0 + (i % 3) as f64, n, 1u16)); }      // bg b1, p~0.06
        for i in 0..8  { data.push((28.0 + i as f64, n, 1u16)); }           // sig b1, p~0.63
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
            "γ_1 should be positive when batch 1 has elevated proportions; got {}",
            fitted.gamma[1]
        );
    }

    #[test]
    fn score_and_decide_matches_assign() {
        let input = make_input_with_totals(
            6, 1,
            vec![(0,0,1),(1,0,1),(2,0,1),(3,0,40),(4,0,42),(5,0,45)],
            vec![50,50,50,50,50,50],
        );
        let model = BinomialModel { min_confidence: 0.8, ..Default::default() };
        let direct = model.assign(&input).unwrap();
        let scores = model.score(&input).unwrap();
        let via_decide = model.decide(&scores, 0.8).unwrap();
        let is_u_d = direct.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let is_u_s = via_decide.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..6 {
            assert_eq!(is_u_d.value(i), is_u_s.value(i), "cell {i}");
        }
    }
}

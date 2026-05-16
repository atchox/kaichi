use super::AssignmentModel;
use super::em::{clamp_probability, log_poisson_pmf, logsumexp2, run_em};
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

#[derive(Clone, Debug)]
struct FitParams {
    pi: f64,
    beta0: f64,
    beta1: f64,
    /// Per-batch log-rate offset. `gamma[0]` is anchored at 0 for identifiability,
    /// so `beta0` is the batch-0 baseline and `gamma[b]` is batch b's offset.
    gamma: Vec<f64>,
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
        let batch = &input.covariates.batch;
        let n_batches = batch.n_batches();

        let log_depths: Vec<f64> = (0..n_cells)
            .map(|i| (total_counts.value(i) as f64 + 1.0).ln())
            .collect();

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
                self.fit_guide(&data, n_batches)
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
        })
    }
}

impl PoissonModel {
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
            for (idx, &(y, log_d, b)) in data.iter().enumerate() {
                let gb = params.gamma[b as usize];
                let mu0 = (params.beta0 + gb + log_d).exp();
                let mu1 = (params.beta0 + params.beta1 + gb + log_d).exp();
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

fn initialize_params(data: &[(f64, f64, u16)], n_batches: usize) -> FitParams {
    let max_y: f64 = data.iter().map(|(y, _, _)| *y).fold(f64::NEG_INFINITY, f64::max);
    let mean_depth: f64 = data.iter().map(|(_, d, _)| d.exp()).sum::<f64>() / data.len() as f64;

    // Per-batch median y → each batch's baseline goes into γ_b. Without this, a global
    // median across heterogeneous batches puts the EM in the wrong basin (the alternating
    // M-step then can't recover because β0 and γ_b are coupled).
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

    // Global max / global median for β1: signal scale shared across batches.
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

    // Reparameterize: η_b = β0 + γ_b (with η_0 = β0). The Hessian of the Poisson
    // log-likelihood in (η_0, ..., η_{B-1}, β1) is arrow-shaped — diagonal in η_b
    // with a single coupling column for β1. We solve it via Schur complement.
    let mut eta: Vec<f64> = (0..n_batches)
        .map(|b| params.beta0 + params.gamma[b])
        .collect();
    let mut beta1 = params.beta1;

    for _ in 0..inner_max_iters {
        // Per-batch score (g_eta[b]) and Hessian diagonal (d[b]),
        // plus the η_b ↔ β1 cross term (c[b]) and the (β1, β1) entry (s).
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

        // Schur complement: eliminate Δη_b from each row, leaving a 1×1 system for Δβ1.
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

        // Back-substitute for each Δη_b.
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

        if max_step < 1e-8 { break; }
    }

    // Recover β0 = η_0 and γ_b = η_b − η_0 (γ_0 ≡ 0 by anchor).
    let beta0 = eta[0];
    let mut gamma = vec![0.0; n_batches];
    for b in 1..n_batches {
        gamma[b] = eta[b] - eta[0];
    }

    // Identifiability: signal mean > background. If the EM landed at β1 < 0 the labels
    // got swapped — relabel by absorbing β1 into β0 and flipping π. (Just negating β1
    // without flipping π would leave π pointing at the low-mean cluster, which is wrong.)
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

    #[test]
    fn n_guides_detected_counts_guides_above_threshold() {
        // Cells 0..3 are the test subjects (0/1/2/3 guides above threshold).
        // Cells 4..7 are extra background ballast so every guide's median count lands
        // firmly in the background regime (β0 fits the small counts, leaving β1
        // free to capture the signal at y ≈ 20).
        let mut triples = vec![
            (0, 0, 1u32), (0, 1, 1), (0, 2, 1),                  // all background
            (1, 0, 20),                                           // 1 signal
            (2, 0, 21), (2, 1, 22),                               // 2 signal
            (3, 0, 20), (3, 1, 21), (3, 2, 22),                   // 3 signal
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

    fn fp(pi: f64, beta0: f64, beta1: f64) -> FitParams {
        FitParams { pi, beta0, beta1, gamma: vec![0.0] }
    }

    #[test]
    fn poisson_posterior_equals_pi_when_components_identical() {
        // β1 = 0 ⇒ μ_signal = μ_bg ⇒ likelihoods identical ⇒ posterior = π.
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
        // log_d=0 → μ_bg=e⁰=1, μ_sig=e³≈20. y=50 is overwhelmingly signal.
        let p = fp(0.2, 0.0, 3.0);
        assert!(posterior_signal(50.0, 0.0, &p, 0) > 0.99);
        assert!(posterior_signal(0.0, 0.0, &p, 0) < 0.2);
    }

    #[test]
    fn poisson_posterior_drops_when_depth_doubles() {
        // log(μ) = β0 + β1·z + log(d+1). Doubling d shifts both μ_bg and μ_sig up by the
        // same factor, so their ratio (and hence β1) is unchanged — but the absolute
        // *difference* μ_sig − μ_bg grows, which makes a fixed y look less surprising
        // under the background model relative to the signal model. Expected outcome:
        // at deeper depth, the same raw count gives a smaller signal posterior.
        let p = fp(0.3, -2.0, 2.0);
        // depth d, log_d = ln(d+1). Compare d=10 vs d=21 (≈ 2× depth in log space).
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
        // Two batches, γ_1 > 0 → μ_bg in batch 1 is e^(γ_1)× larger than in batch 0.
        // Same y at log_d=0 should give a LOWER signal posterior in batch 1 (the count
        // looks more like background in a batch with higher baseline rate).
        let p = FitParams { pi: 0.3, beta0: -2.0, beta1: 2.0, gamma: vec![0.0, 1.5] };
        let post_b0 = posterior_signal(5.0, 0.0, &p, 0);
        let post_b1 = posterior_signal(5.0, 0.0, &p, 1);
        assert!(post_b0 > post_b1, "γ_1 > 0 should reduce signal posterior in batch 1: b0={post_b0}, b1={post_b1}");
    }

    // ---- fit_mixture: synthetic recovery ----

    #[test]
    fn fit_mixture_recovers_bimodal_poisson() {
        // 12 background (counts 0–2) + 8 signal (counts 18–25), uniform depth 50, single batch.
        let log_d = (50.0_f64 + 1.0).ln();
        let mut data: Vec<(f64, f64, u16)> = (0..12).map(|i| ((i % 3) as f64, log_d, 0u16)).collect();
        data.extend((0..8).map(|i| (18.0 + i as f64, log_d, 0u16)));

        let fitted = fit_mixture(&data, 1, 200, 25, 1e-8);
        let mu_bg = (fitted.beta0 + log_d).exp();
        let mu_sig = (fitted.beta0 + fitted.beta1 + log_d).exp();

        assert!(mu_sig > mu_bg, "μ_sig={mu_sig} ≤ μ_bg={mu_bg}");
        assert!(mu_bg < 5.0, "μ_bg should track low cluster, got {mu_bg}");
        assert!(mu_sig > 15.0, "μ_sig should track high cluster, got {mu_sig}");
        assert!((0.2..=0.6).contains(&fitted.pi), "pi: {}", fitted.pi);
        assert_eq!(fitted.gamma.len(), 1);
        assert_eq!(fitted.gamma[0], 0.0, "γ_0 must stay anchored at 0");
    }

    #[test]
    fn fit_mixture_recovers_batch_offsets() {
        // Two batches with the SAME signal/background structure but DIFFERENT baselines.
        // Batch 0: bg ≈ 1, sig ≈ 20  (β0 ≈ -3.9 at log_d≈3.9, so μ_bg = 1.)
        // Batch 1: bg ≈ 4, sig ≈ 80  (γ_1 ≈ ln(4) ≈ 1.39 above batch 0.)
        // After fit, γ_1 > 0 and roughly recovers the batch-1 baseline offset.
        let log_d = (50.0_f64 + 1.0).ln();
        let mut data: Vec<(f64, f64, u16)> = Vec::new();
        // batch 0: 10 bg + 6 sig
        for i in 0..10 { data.push(((i % 2 + 1) as f64, log_d, 0u16)); }       // y = 1 or 2
        for i in 0..6  { data.push((18.0 + i as f64, log_d, 0u16)); }          // y ≈ 20
        // batch 1: 10 bg + 6 sig at 4× the rate
        for i in 0..10 { data.push(((i % 2 + 4) as f64, log_d, 1u16)); }       // y = 4 or 5
        for i in 0..6  { data.push((75.0 + i as f64, log_d, 1u16)); }          // y ≈ 80

        let fitted = fit_mixture(&data, 2, 200, 50, 1e-8);
        assert!(fitted.gamma[0] == 0.0, "γ_0 anchored");
        // Truth γ_1 = ln(4) ≈ 1.386; with this synthetic data the EM lands near 1.307.
        // 0.15 window catches a 2× drift from the current fit while remaining tight enough
        // to flag a real regression.
        assert!(
            (fitted.gamma[1] - 4.0_f64.ln()).abs() < 0.15,
            "γ_1 expected ≈ ln(4)={:.3}, got {:.3}", 4.0_f64.ln(), fitted.gamma[1]
        );
    }
}

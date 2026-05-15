use super::AssignmentModel;
use super::em::{clamp_probability, log_beta_pdf, logsumexp2, run_em};
use super::output::AssignmentOutputBuilder;
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Beta2
// ---------------------------------------------------------------------------

/// 2-component Beta mixture on the per-cell maximum guide proportion.
///
/// x_i = max_g(y_{ig} / Σ_{g'} y_{ig'}).  x_i ~ π·Beta(α_h,β_h) + (1-π)·Beta(α_l,β_l).
/// The signal component (h) has mean close to 1; the background (l) close to 0.
/// M-step uses method-of-moments (closed form). Single-batch.
pub struct Beta2Model {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub tol: f32,
    pub clamp_lo: f32,
    pub clamp_hi: f32,
}

impl Default for Beta2Model {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_em_iters: 200,
            tol: 1e-6,
            clamp_lo: 1e-4,
            clamp_hi: 1.0 - 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// Beta3
// ---------------------------------------------------------------------------

/// 3-component Beta mixture on the per-cell maximum guide proportion.
///
/// Same as Beta2 but adds an intermediate component (α_m, β_m).
/// Only the high component (h) is treated as signal for assignment.
pub struct Beta3Model {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub tol: f32,
    pub clamp_lo: f32,
    pub clamp_hi: f32,
}

impl Default for Beta3Model {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_em_iters: 200,
            tol: 1e-6,
            clamp_lo: 1e-4,
            clamp_hi: 1.0 - 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Beta2Params {
    pi: f64,         // weight of high (signal) component
    al: f64, bl: f64, // background Beta(α_l, β_l)
    ah: f64, bh: f64, // signal Beta(α_h, β_h)
}

#[derive(Clone, Copy, Debug)]
struct Beta3Params {
    pi_l: f64, pi_m: f64, pi_h: f64,
    al: f64, bl: f64,
    am: f64, bm: f64,
    ah: f64, bh: f64,
}

// ---------------------------------------------------------------------------
// AssignmentModel impls
// ---------------------------------------------------------------------------

impl AssignmentModel for Beta2Model {
    fn name(&self) -> &'static str { "beta2" }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csr = input.counts.csr();
        let total_counts = &input.covariates.total_counts;

        let props = max_proportions(
            csr, n_cells, total_counts,
            self.clamp_lo as f64, self.clamp_hi as f64,
        );

        let fitted = fit_beta2(
            &props.iter().filter_map(|&(p, _, _)| if p > 0.0 { Some(p) } else { None }).collect::<Vec<_>>(),
            self.max_em_iters,
            self.tol as f64,
        );

        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let (prop, guide_idx, count) = props[cell];
            if prop <= 0.0 {
                out.append_unassigned(0);
                continue;
            }
            let post = posterior_high_2(prop, fitted) as f32;
            if post >= self.min_confidence {
                out.append_assigned(guide_ids_arr.value(guide_idx), count, post, false, 1);
                out.push_assigned_triple(cell, guide_idx);
            } else {
                out.append_unassigned(0);
            }
        }

        out.finish(cell_barcodes, false)
    }

    fn params_json(&self) -> Value {
        json!({
            "min_confidence": self.min_confidence,
            "max_em_iters": self.max_em_iters,
            "tol": self.tol,
            "clamp_lo": self.clamp_lo,
            "clamp_hi": self.clamp_hi,
        })
    }
}

impl AssignmentModel for Beta3Model {
    fn name(&self) -> &'static str { "beta3" }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csr = input.counts.csr();
        let total_counts = &input.covariates.total_counts;

        let props = max_proportions(
            csr, n_cells, total_counts,
            self.clamp_lo as f64, self.clamp_hi as f64,
        );

        let fitted = fit_beta3(
            &props.iter().filter_map(|&(p, _, _)| if p > 0.0 { Some(p) } else { None }).collect::<Vec<_>>(),
            self.max_em_iters,
            self.tol as f64,
        );

        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let (prop, guide_idx, count) = props[cell];
            if prop <= 0.0 {
                out.append_unassigned(0);
                continue;
            }
            // Only the high (h) component is treated as signal.
            let post = posterior_high_3(prop, fitted) as f32;
            if post >= self.min_confidence {
                out.append_assigned(guide_ids_arr.value(guide_idx), count, post, false, 1);
                out.push_assigned_triple(cell, guide_idx);
            } else {
                out.append_unassigned(0);
            }
        }

        out.finish(cell_barcodes, false)
    }

    fn params_json(&self) -> Value {
        json!({
            "min_confidence": self.min_confidence,
            "max_em_iters": self.max_em_iters,
            "tol": self.tol,
            "clamp_lo": self.clamp_lo,
            "clamp_hi": self.clamp_hi,
        })
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Per-cell maximum guide proportion (proportion, guide_idx, raw_count).
/// Returns (0.0, 0, 0) for cells with no guide UMIs.
fn max_proportions(
    csr: &nalgebra_sparse::CsrMatrix<u32>,
    n_cells: usize,
    total_counts: &arrow::array::Float32Array,
    clamp_lo: f64,
    clamp_hi: f64,
) -> Vec<(f64, usize, u32)> {
    let mut out = Vec::with_capacity(n_cells);
    for cell in 0..n_cells {
        let total = total_counts.value(cell) as f64;
        if total <= 0.0 {
            out.push((0.0, 0, 0));
            continue;
        }
        let row = csr.get_row(cell).unwrap();
        let mut best_prop = 0.0;
        let mut best_guide = 0;
        let mut best_count = 0u32;
        for (&g, &v) in row.col_indices().iter().zip(row.values()) {
            let p = v as f64 / total;
            if p > best_prop {
                best_prop = p;
                best_guide = g;
                best_count = v;
            }
        }
        out.push((best_prop.clamp(clamp_lo, clamp_hi), best_guide, best_count));
    }
    out
}

// ---------------------------------------------------------------------------
// Beta2 EM
// ---------------------------------------------------------------------------

fn fit_beta2(props: &[f64], max_em_iters: u32, tol: f64) -> Beta2Params {
    let init = Beta2Params { pi: 0.4, al: 1.0, bl: 10.0, ah: 10.0, bh: 1.0 };
    let mut r_h = vec![0.0f64; props.len()];

    run_em(
        init,
        |params| {
            let mut log_lik = 0.0;
            for (idx, &x) in props.iter().enumerate() {
                let log_l = (1.0 - params.pi).ln() + log_beta_pdf(x, params.al, params.bl);
                let log_h = params.pi.ln() + log_beta_pdf(x, params.ah, params.bh);
                let denom = logsumexp2(log_l, log_h);
                r_h[idx] = (log_h - denom).exp();
                log_lik += denom;
            }
            let new_params = m_step_beta2(params, props, &r_h);
            (new_params, log_lik)
        },
        max_em_iters,
        tol,
    )
}

fn m_step_beta2(_params: Beta2Params, props: &[f64], r_h: &[f64]) -> Beta2Params {
    // pi = weight of the HIGH component = E[z_i = high].
    let pi = clamp_probability(r_h.iter().sum::<f64>() / props.len() as f64);

    // Method of moments per component.
    let (al, bl) = beta_mom_weighted(props, r_h, false);
    let (ah, bh) = beta_mom_weighted(props, r_h, true);

    // Sort: ensure l component has lower mean than h.
    // When components swap labels, pi (the weight of the HIGH component) must also swap.
    let mean_l = al / (al + bl);
    let mean_h = ah / (ah + bh);
    if mean_l <= mean_h {
        Beta2Params { pi, al, bl, ah, bh }
    } else {
        Beta2Params { pi: 1.0 - pi, al: ah, bl: bh, ah: al, bh: bl }
    }
}

fn posterior_high_2(x: f64, params: Beta2Params) -> f64 {
    let log_l = (1.0 - params.pi).ln() + log_beta_pdf(x, params.al, params.bl);
    let log_h = params.pi.ln() + log_beta_pdf(x, params.ah, params.bh);
    let denom = logsumexp2(log_l, log_h);
    (log_h - denom).exp()
}

// ---------------------------------------------------------------------------
// Beta3 EM
// ---------------------------------------------------------------------------

fn fit_beta3(props: &[f64], max_em_iters: u32, tol: f64) -> Beta3Params {
    let init = Beta3Params {
        pi_l: 0.4, pi_m: 0.3, pi_h: 0.3,
        al: 1.0, bl: 10.0,
        am: 10.0, bm: 10.0,
        ah: 10.0, bh: 1.0,
    };
    let n = props.len();
    let mut r = vec![[0.0f64; 3]; n]; // [r_l, r_m, r_h]

    run_em(
        init,
        |params| {
            let mut log_lik = 0.0;
            for (idx, &x) in props.iter().enumerate() {
                let ll = params.pi_l.ln() + log_beta_pdf(x, params.al, params.bl);
                let lm = params.pi_m.ln() + log_beta_pdf(x, params.am, params.bm);
                let lh = params.pi_h.ln() + log_beta_pdf(x, params.ah, params.bh);
                let max = ll.max(lm).max(lh);
                let sum_exp = (ll - max).exp() + (lm - max).exp() + (lh - max).exp();
                let denom = max + sum_exp.ln();
                r[idx][0] = (ll - denom).exp();
                r[idx][1] = (lm - denom).exp();
                r[idx][2] = (lh - denom).exp();
                log_lik += denom;
            }
            let new_params = m_step_beta3(params, props, &r);
            (new_params, log_lik)
        },
        max_em_iters,
        tol,
    )
}

fn m_step_beta3(_params: Beta3Params, props: &[f64], r: &[[f64; 3]]) -> Beta3Params {
    let n = props.len() as f64;
    let sum_rl: f64 = r.iter().map(|ri| ri[0]).sum();
    let sum_rm: f64 = r.iter().map(|ri| ri[1]).sum();
    let sum_rh: f64 = r.iter().map(|ri| ri[2]).sum();

    let pi_l = clamp_probability(sum_rl / n);
    let pi_m = clamp_probability(sum_rm / n);
    let pi_h = clamp_probability(sum_rh / n);

    let r_l: Vec<f64> = r.iter().map(|ri| ri[0]).collect();
    let r_m: Vec<f64> = r.iter().map(|ri| ri[1]).collect();
    let r_h: Vec<f64> = r.iter().map(|ri| ri[2]).collect();

    let (al, bl) = beta_mom_weighted(props, &r_l, true);
    let (am, bm) = beta_mom_weighted(props, &r_m, true);
    let (ah, bh) = beta_mom_weighted(props, &r_h, true);

    // Sort components by mean so l < m < h stays labeled correctly.
    let mut comps = [(al, bl, pi_l), (am, bm, pi_m), (ah, bh, pi_h)];
    comps.sort_by(|(a0, b0, _), (a1, b1, _)| {
        (a0 / (a0 + b0)).partial_cmp(&(a1 / (a1 + b1))).unwrap()
    });

    Beta3Params {
        pi_l: comps[0].2, al: comps[0].0, bl: comps[0].1,
        pi_m: comps[1].2, am: comps[1].0, bm: comps[1].1,
        pi_h: comps[2].2, ah: comps[2].0, bh: comps[2].1,
    }
}

fn posterior_high_3(x: f64, params: Beta3Params) -> f64 {
    let ll = params.pi_l.ln() + log_beta_pdf(x, params.al, params.bl);
    let lm = params.pi_m.ln() + log_beta_pdf(x, params.am, params.bm);
    let lh = params.pi_h.ln() + log_beta_pdf(x, params.ah, params.bh);
    let max = ll.max(lm).max(lh);
    let sum_exp = (ll - max).exp() + (lm - max).exp() + (lh - max).exp();
    (lh - max).exp() / sum_exp
}

// ---------------------------------------------------------------------------
// Method-of-moments Beta parameter estimation
// ---------------------------------------------------------------------------

/// Method-of-moments Beta(α, β) from weighted proportions.
/// `signal_weight`: if true, use weights as-is; if false, use 1-weights.
fn beta_mom_weighted(props: &[f64], weights: &[f64], signal_weight: bool) -> (f64, f64) {
    let w: Vec<f64> = if signal_weight {
        weights.to_vec()
    } else {
        weights.iter().map(|&r| 1.0 - r).collect()
    };
    let sum_w: f64 = w.iter().sum();
    if sum_w < 1e-6 {
        return (1.0, 1.0);
    }
    let mean: f64 = props.iter().zip(&w).map(|(&x, &wi)| wi * x).sum::<f64>() / sum_w;
    let var: f64 = props.iter().zip(&w).map(|(&x, &wi)| wi * (x - mean).powi(2)).sum::<f64>() / sum_w;
    if var <= 0.0 || mean <= 0.0 || mean >= 1.0 {
        // Degenerate: fall back to a reasonable fixed Beta.
        let phi = if signal_weight { 10.0 } else { 1.0 };
        return (mean.max(0.01) * phi, (1.0 - mean.max(0.01).min(0.99)) * phi);
    }
    let phi = (mean * (1.0 - mean) / var - 1.0).max(0.1);
    (mean * phi, (1.0 - mean) * phi)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_support::{input_with_row_sums as make_input, input_with_totals};
    use crate::data::CountMatrix;
    use arrow::array::{BooleanArray, Float32Array};

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    fn make_input_with_totals(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        totals: Vec<u32>,
    ) -> LoadedInput {
        input_with_totals(n_cells, n_guides, triples,
            totals.iter().map(|&t| t as f32).collect())
    }

    #[test]
    fn m_step_beta2_swaps_pi_with_components() {
        // Construct responsibilities where r_h is assigned to the LOW-proportion data
        // (components need to swap). After the swap, pi should equal 1 - original_pi.
        let props: Vec<f64> = vec![0.1, 0.1, 0.1, 0.9, 0.9];
        // r_h = responsibility for the "high" label — but let's deliberately assign
        // high responsibility to the low-proportion cells so components must swap.
        let r_h: Vec<f64> = vec![0.9, 0.9, 0.9, 0.1, 0.1];

        // Before the m_step call, "pi" from r_h.sum/n = 0.9*3 + 0.1*2 = 2.9 / 5 = 0.58.
        // After MOM, the component fit to r_h weights has low mean (≈ 0.1) and the
        // component fit to (1-r_h) weights has high mean (≈ 0.9). Labels must swap.
        // pi after swap must be 1.0 - 0.58 = 0.42.
        let dummy_params = Beta2Params { pi: 0.5, al: 1.0, bl: 10.0, ah: 10.0, bh: 1.0 };
        let result = m_step_beta2(dummy_params, &props, &r_h);

        assert!(result.ah / (result.ah + result.bh) > result.al / (result.al + result.bl),
            "high component should have higher mean than low after m_step");
        // When components swap, pi should be approximately 1 - (r_h.sum/n).
        let original_pi = r_h.iter().sum::<f64>() / r_h.len() as f64;
        let expected_pi_after_swap = 1.0 - original_pi;
        assert!((result.pi - expected_pi_after_swap).abs() < 1e-6,
            "pi should be swapped: expected {expected_pi_after_swap:.4}, got {:.4}", result.pi);
    }

    #[test]
    fn beta2_assigns_dominant_guide_cells() {
        // Background: max_prop ≈ 1/5 = 0.2 (5 equal guides, 2 UMI each).
        // Signal: max_prop = 8/10 = 0.8 (guide 0 dominates).
        let n_cells = 8;
        // Background cells (0-3): 5 guides with 2 UMI each (total=10, max_prop=0.2).
        // Signal cells (4-7): guide 0 = 8 UMI, others = 0 (total=10, max_prop=0.8).
        let triples: Vec<(usize, usize, u32)> = (0..4)
            .flat_map(|c| (0..5).map(move |g| (c, g, 2u32)))
            .chain((4..8).map(|c| (c, 0, 8u32)))
            .collect();
        let totals = vec![10u32; n_cells];
        let input = make_input_with_totals(n_cells, 5, triples, totals);
        let model = Beta2Model { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        for i in 0..4 { assert!(is_u.value(i), "background cell {i} should be unassigned"); }
        for i in 4..8 { assert!(!is_u.value(i), "signal cell {i} should be assigned"); }
    }

    #[test]
    fn beta3_assigns_dominant_guide_cells() {
        // Same setup as beta2 but with beta3.
        let n_cells = 8;
        let triples: Vec<(usize, usize, u32)> = (0..4)
            .flat_map(|c| (0..5).map(move |g| (c, g, 2u32)))
            .chain((4..8).map(|c| (c, 0, 8u32)))
            .collect();
        let totals = vec![10u32; n_cells];
        let input = make_input_with_totals(n_cells, 5, triples, totals);
        let model = Beta3Model { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        for i in 0..4 { assert!(is_u.value(i), "background cell {i} should be unassigned"); }
        for i in 4..8 { assert!(!is_u.value(i), "signal cell {i} should be assigned"); }
    }

    #[test]
    fn beta2_zero_total_is_unassigned() {
        let input = make_input(3, 2, vec![]); // all cells have 0 total counts
        let result = Beta2Model::default().assign(&input).unwrap();
        for i in 0..3 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn beta2_empty_input() {
        let input = make_input(0, 0, vec![]);
        let result = Beta2Model::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn beta3_empty_input() {
        let input = make_input(0, 0, vec![]);
        let result = Beta3Model::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn unassigned_columns_are_null_beta2() {
        use arrow::array::Array;
        let input = make_input(3, 2, vec![]);
        let result = Beta2Model::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
        }
    }

    #[test]
    fn beta3_zero_total_is_unassigned() {
        let input = make_input(3, 2, vec![]);
        let result = Beta3Model::default().assign(&input).unwrap();
        for i in 0..3 { assert!(is_unassigned(&result).value(i)); }
    }

    #[test]
    fn unassigned_columns_are_null_beta3() {
        use arrow::array::Array;
        let input = make_input(3, 2, vec![]);
        let result = Beta3Model::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
        }
    }

    // ---- beta_mom_weighted (the moment-match identity) ----

    #[test]
    fn mom_recovers_mean_and_variance_at_uniform_weights() {
        // Identity: for Beta(α, β) the relations
        //   mean = α/(α+β),  var = α·β / [(α+β)²·(α+β+1)]
        // are equivalent to  φ = α+β = mean·(1-mean)/var - 1.
        // With uniform weights, MOM must round-trip mean and var exactly.
        let xs = [0.05, 0.10, 0.20, 0.35, 0.55, 0.80, 0.95];
        let w = vec![1.0; xs.len()];

        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / xs.len() as f64;

        let (a, b) = beta_mom_weighted(&xs, &w, true);

        let recovered_mean = a / (a + b);
        let recovered_var = (a * b) / ((a + b).powi(2) * (a + b + 1.0));

        assert!((recovered_mean - mean).abs() < 1e-12, "mean: got {recovered_mean}, want {mean}");
        assert!((recovered_var - var).abs() < 1e-12, "var: got {recovered_var}, want {var}");
    }

    #[test]
    fn mom_signal_weight_flag_inverts_weights() {
        // Passing signal_weight=false should fit the SAME (α, β) we'd get by passing
        // signal_weight=true with explicitly inverted weights.
        let xs = [0.1, 0.2, 0.6, 0.8, 0.9];
        let r = [0.9, 0.7, 0.5, 0.3, 0.1];
        let r_inv: Vec<f64> = r.iter().map(|&v| 1.0 - v).collect();

        let (a1, b1) = beta_mom_weighted(&xs, &r, false);
        let (a2, b2) = beta_mom_weighted(&xs, &r_inv, true);

        assert!((a1 - a2).abs() < 1e-12 && (b1 - b2).abs() < 1e-12);
    }

    #[test]
    fn mom_returns_fallback_when_weights_vanish() {
        // sum_w < 1e-6 → uninformative Beta(1, 1).
        let xs = [0.3, 0.7];
        let zero = [0.0, 0.0];
        let (a, b) = beta_mom_weighted(&xs, &zero, true);
        assert_eq!((a, b), (1.0, 1.0));
    }

    #[test]
    fn mom_handles_degenerate_zero_variance() {
        // All observations equal → var=0 → degenerate branch must not produce NaN/Inf.
        let xs = [0.4, 0.4, 0.4];
        let w = [1.0, 1.0, 1.0];
        let (a, b) = beta_mom_weighted(&xs, &w, true);
        assert!(a.is_finite() && b.is_finite() && a > 0.0 && b > 0.0);
    }

    // ---- max_proportions ----

    fn csr_from_triples(n_cells: usize, n_guides: usize, triples: &[(usize, usize, u32)])
        -> nalgebra_sparse::CsrMatrix<u32>
    {
        let mut sorted = triples.to_vec();
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());
        let mut last = 0usize;
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r { row_offsets[last + 1] = idx; last += 1; }
            col_indices.push(c);
            values.push(v);
        }
        for i in (last + 1)..=n_cells { row_offsets[i] = sorted.len(); }
        let counts = CountMatrix::try_from_csr(
            n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        counts.csr().clone()
    }

    #[test]
    fn max_proportions_picks_largest_guide() {
        // Cell 0: g0=2, g1=8 → max_prop = 8/10 = 0.8, guide = 1, count = 8.
        let csr = csr_from_triples(1, 2, &[(0, 0, 2), (0, 1, 8)]);
        let totals = Float32Array::from(vec![10.0]);
        let props = max_proportions(&csr, 1, &totals, 1e-4, 1.0 - 1e-4);
        assert_eq!(props.len(), 1);
        let (p, g, c) = props[0];
        assert!((p - 0.8).abs() < 1e-12);
        assert_eq!(g, 1);
        assert_eq!(c, 8);
    }

    #[test]
    fn max_proportions_zero_total_yields_zero_tuple() {
        // total_counts = 0 → returns (0.0, 0, 0) regardless of sparse content.
        let csr = csr_from_triples(2, 2, &[]);
        let totals = Float32Array::from(vec![0.0, 0.0]);
        let props = max_proportions(&csr, 2, &totals, 1e-4, 1.0 - 1e-4);
        for (p, g, c) in &props {
            assert_eq!((*p, *g, *c), (0.0, 0, 0));
        }
    }

    #[test]
    fn max_proportions_clamps_to_bounds() {
        // p = 1.0 exactly clamps down to clamp_hi (Beta pdf undefined at 1.0).
        let csr = csr_from_triples(1, 1, &[(0, 0, 5)]);
        let totals = Float32Array::from(vec![5.0]);
        let props = max_proportions(&csr, 1, &totals, 1e-4, 0.9999);
        assert!((props[0].0 - 0.9999).abs() < 1e-12);
    }

    #[test]
    fn max_proportions_ties_keep_first_seen() {
        // Strict `>` in the inner loop → first guide seen at the max wins.
        let csr = csr_from_triples(1, 2, &[(0, 0, 5), (0, 1, 5)]);
        let totals = Float32Array::from(vec![10.0]);
        let props = max_proportions(&csr, 1, &totals, 1e-4, 0.9999);
        assert_eq!(props[0].1, 0, "tie should resolve to lowest guide index");
    }

    // ---- posterior_high_2 ----

    #[test]
    fn posterior_equals_pi_when_components_identical() {
        // Identical components ⇒ likelihood ratio = 1 ⇒ posterior = π.
        for &pi in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let p = Beta2Params { pi, al: 5.0, bl: 5.0, ah: 5.0, bh: 5.0 };
            let post = posterior_high_2(0.42, p);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }

    #[test]
    fn posterior_saturates_with_extreme_pi() {
        // pi just above the clamp floor → posterior pinned near 0; just below the cap → near 1.
        // (We use clamp bounds 1e-6 because the EM clamps pi via clamp_probability.)
        let p_lo = Beta2Params { pi: 1e-6, al: 1.0, bl: 10.0, ah: 10.0, bh: 1.0 };
        let p_hi = Beta2Params { pi: 1.0 - 1e-6, al: 1.0, bl: 10.0, ah: 10.0, bh: 1.0 };
        assert!(posterior_high_2(0.5, p_lo) < 1e-5);
        assert!(posterior_high_2(0.5, p_hi) > 1.0 - 1e-5);
    }

    #[test]
    fn posterior_monotonic_when_signal_mean_higher() {
        // Background Beta(1, 10) (mean ≈ 0.09), signal Beta(10, 1) (mean ≈ 0.91), pi=0.4.
        // posterior_high_2 must be monotonically increasing in x.
        let p = Beta2Params { pi: 0.4, al: 1.0, bl: 10.0, ah: 10.0, bh: 1.0 };
        let xs = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95];
        let mut prev = -1.0;
        for x in xs {
            let post = posterior_high_2(x, p);
            assert!(post > prev, "non-monotonic at x={x}: prev={prev}, post={post}");
            prev = post;
        }
    }

    #[test]
    fn posterior_crossover_at_half_when_weights_equal() {
        // π=0.5 with mirrored Beta(α,β) and Beta(β,α): equal density at x=0.5 → post = 0.5.
        let p = Beta2Params { pi: 0.5, al: 3.0, bl: 8.0, ah: 8.0, bh: 3.0 };
        assert!((posterior_high_2(0.5, p) - 0.5).abs() < 1e-12);
    }

    // ---- m_step_beta2 (perfect-responsibility recovery) ----

    #[test]
    fn m_step_recovers_two_clusters_with_perfect_responsibilities() {
        // Two clusters at means 0.2 and 0.8, equal sizes, hard assignment.
        // MOM on each half should give means 0.2 and 0.8 (within tolerance).
        let lo = vec![0.15, 0.20, 0.25];
        let hi = vec![0.75, 0.80, 0.85];
        let props: Vec<f64> = lo.iter().chain(hi.iter()).copied().collect();
        // r_h = 1 for hi cluster, 0 for lo cluster.
        let r_h: Vec<f64> = lo.iter().map(|_| 0.0).chain(hi.iter().map(|_| 1.0)).collect();

        let dummy = Beta2Params { pi: 0.5, al: 1.0, bl: 1.0, ah: 1.0, bh: 1.0 };
        let out = m_step_beta2(dummy, &props, &r_h);

        let mean_l = out.al / (out.al + out.bl);
        let mean_h = out.ah / (out.ah + out.bh);
        assert!((mean_l - 0.20).abs() < 1e-12, "mean_l = {mean_l}");
        assert!((mean_h - 0.80).abs() < 1e-12, "mean_h = {mean_h}");
        assert!((out.pi - 0.5).abs() < 1e-12, "pi = {}", out.pi);
    }

    // ---- fit_beta2 end-to-end on synthetic data ----

    fn sample_beta(alpha: f64, beta_: f64, n: usize, seed: u64) -> Vec<f64> {
        // Mersenne-twister-free deterministic Beta sampler via inverse-CDF on a LCG.
        // Adequate for tests where we only check approximate recovery, not draw fidelity.
        // Use the gamma-ratio identity: X ~ Beta(α, β) iff X = G_α/(G_α + G_β), G ~ Gamma.
        // Approximate Gamma via the Marsaglia–Tsang transform isn't worth the complexity;
        // instead we use a coarse but adequate transform: sum of -log(U)/α for shape α.
        // (Valid only for shape ≥ 1; we pick α, β ≥ 1 in the test that uses this.)
        use std::num::Wrapping;
        let mut state = Wrapping(seed.max(1));
        let mut next = || {
            state = state * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
            // uniform in (0, 1)
            ((state.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        };
        let gamma = |shape: f64, draw: &mut dyn FnMut() -> f64| -> f64 {
            // sum of `shape` exponentials, fractional part via Whittaker — coarse.
            let k = shape.floor() as usize;
            let mut s = 0.0;
            for _ in 0..k { s += -draw().ln(); }
            s
        };
        (0..n).map(|_| {
            let ga = gamma(alpha, &mut next);
            let gb = gamma(beta_, &mut next);
            ga / (ga + gb)
        }).collect()
    }

    #[test]
    fn fit_beta2_recovers_well_separated_clusters() {
        // Mix 70% Beta(2, 18) (mean 0.1) + 30% Beta(18, 2) (mean 0.9).
        // After EM, signal component (higher mean) should land near 0.9.
        let mut data = sample_beta(2.0, 18.0, 700, 42);
        data.extend(sample_beta(18.0, 2.0, 300, 7));
        // Clamp away from the boundaries the model expects.
        for x in &mut data { *x = x.clamp(1e-4, 1.0 - 1e-4); }

        let fitted = fit_beta2(&data, 500, 1e-8);
        let mean_h = fitted.ah / (fitted.ah + fitted.bh);
        let mean_l = fitted.al / (fitted.al + fitted.bl);

        assert!(mean_h > mean_l, "signal mean must exceed background mean");
        assert!((mean_h - 0.9).abs() < 0.05, "mean_h drifted: {mean_h}");
        assert!((mean_l - 0.1).abs() < 0.05, "mean_l drifted: {mean_l}");
        // 30% signal weight (give EM some slack).
        assert!((fitted.pi - 0.3).abs() < 0.10, "pi drifted: {}", fitted.pi);
    }
}

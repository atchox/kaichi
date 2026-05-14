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

fn m_step_beta2(params: Beta2Params, props: &[f64], r_h: &[f64]) -> Beta2Params {
    let pi = clamp_probability(r_h.iter().sum::<f64>() / props.len() as f64);

    // Method of moments per component.
    let (al, bl) = beta_mom_weighted(props, r_h, false);
    let (ah, bh) = beta_mom_weighted(props, r_h, true);

    // Sort: ensure l component has lower mean than h.
    let mean_l = al / (al + bl);
    let mean_h = ah / (ah + bh);
    if mean_l <= mean_h {
        Beta2Params { pi, al, bl, ah, bh }
    } else {
        Beta2Params { pi, al: ah, bl: bh, ah: al, bh: bl }
    }
    // If components swap, also swap pi (but use the current pi as is; it'll re-sort next iter).
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

fn m_step_beta3(params: Beta3Params, props: &[f64], r: &[[f64; 3]]) -> Beta3Params {
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
    let phi = mean * (1.0 - mean) / var - 1.0;
    if phi <= 0.0 {
        let phi = 1.0;
        return (mean * phi, (1.0 - mean) * phi);
    }
    (mean * phi, (1.0 - mean) * phi)
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

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    // Build input with explicit per-cell total_counts (for multi-guide cells where
    // one guide dominates the proportion).
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
}

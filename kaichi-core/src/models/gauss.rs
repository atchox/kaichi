use super::AssignmentModel;
use super::em::{clamp_probability, log_normal_pdf, logsumexp2, run_em};
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};

/// Assign each cell to a guide via a two-component tied-variance Gaussian mixture
/// fitted on log10(UMI + 1) per guide.
///
/// Background: N(μ_bg, σ²). Signal: N(μ_signal, σ²). Shared σ (tied variance).
/// Every cell contributes to the EM (zeros as log10(1) = 0.0, handled analytically).
/// A raw-count threshold is derived as the first count whose signal posterior exceeds
/// `min_confidence`.
pub struct GaussModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
}

impl Default for GaussModel {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_em_iters: 100,
            tol: 1e-6,
            min_nonzero: 2,
            min_max_count: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FitParams {
    pi: f64,
    mu_bg: f64,
    mu_signal: f64,
    sigma: f64,
}

impl AssignmentModel for GaussModel {
    fn name(&self) -> &'static str {
        "gauss"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();

        // Step 1: per-guide EM in parallel via CSC columns.
        let guide_fits: Vec<Option<(u32, FitParams)>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let n_zeros = n_cells - col.nnz();
                self.fit_guide(col.values(), n_zeros)
            })
            .collect();

        // Step 2: per-cell scoring via CSR.
        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;
        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let row = csr.get_row(cell).unwrap();
            let mut best: Option<(usize, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values().iter()) {
                let Some((threshold, params)) = guide_fits[guide_idx] else { continue };
                if count < threshold {
                    continue;
                }
                let x = (count as f64 + 1.0).log10();
                let posterior = posterior_signal(x, params) as f32;
                if posterior >= self.min_confidence {
                    n_passing += 1;
                    match best {
                        None => best = Some((guide_idx, posterior, count)),
                        Some((_, bp, bc)) => {
                            if posterior > bp || (posterior == bp && count > bc) {
                                best = Some((guide_idx, posterior, count));
                            }
                        }
                    }
                    out.push_assigned_triple(cell, guide_idx);
                }
            }

            let n_det = n_detected_u8(n_passing);
            match best {
                None => out.append_unassigned(n_det),
                Some((guide_idx, posterior, count)) => {
                    out.append_assigned(
                        guide_ids_arr.value(guide_idx),
                        count,
                        posterior,
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
            "tol": self.tol,
            "min_nonzero": self.min_nonzero,
            "min_max_count": self.min_max_count,
        })
    }
}

impl GaussModel {
    fn fit_guide(&self, nonzero_counts: &[u32], n_zeros: usize) -> Option<(u32, FitParams)> {
        let n_nonzero = nonzero_counts.iter().filter(|&&c| c > 0).count() as u32;
        let max_count = nonzero_counts.iter().copied().max().unwrap_or(0);

        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }

        let log_vals: Vec<f64> = nonzero_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| (c as f64 + 1.0).log10())
            .collect();

        let params = fit_mixture(&log_vals, n_zeros, self.max_em_iters, self.tol as f64);

        let threshold = (1..=max_count)
            .find(|&c| posterior_signal((c as f64 + 1.0).log10(), params) > self.min_confidence as f64)?;

        Some((threshold, params))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

/// EM on a tied-variance two-component Gaussian mixture over log10(UMI + 1).
///
/// `log_vals`: log10(count + 1) for nonzero UMI observations.
/// `n_zeros`: cells with zero counts (log10(1) = 0.0), handled analytically.
fn fit_mixture(log_vals: &[f64], n_zeros: usize, max_em_iters: u32, tol: f64) -> FitParams {
    let init = initialize_params(log_vals, n_zeros);
    let mut responsibilities = vec![0.0f64; log_vals.len()];

    run_em(
        init,
        |params| {
            let mut log_lik = 0.0;
            for (idx, &x) in log_vals.iter().enumerate() {
                let log_bg = (1.0 - params.pi).ln() + log_normal_pdf(x, params.mu_bg, params.sigma);
                let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma);
                let denom = logsumexp2(log_bg, log_sig);
                responsibilities[idx] = (log_sig - denom).exp();
                log_lik += denom;
            }
            // Zeros: log10(0 + 1) = 0.0; all identical, one shared responsibility.
            let log_bg_z = (1.0 - params.pi).ln() + log_normal_pdf(0.0, params.mu_bg, params.sigma);
            let log_sig_z = params.pi.ln() + log_normal_pdf(0.0, params.mu_signal, params.sigma);
            let denom_z = logsumexp2(log_bg_z, log_sig_z);
            let r_zero = (log_sig_z - denom_z).exp();
            log_lik += n_zeros as f64 * denom_z;
            let new_params = m_step(log_vals, &responsibilities, n_zeros, r_zero);
            (new_params, log_lik)
        },
        max_em_iters,
        tol,
    )
}

fn initialize_params(log_vals: &[f64], n_zeros: usize) -> FitParams {
    let mut sorted = log_vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let threshold_idx = ((sorted.len() as f64) * 0.9).floor() as usize;
    let threshold = sorted[threshold_idx.min(sorted.len() - 1)];
    let resp: Vec<f64> = log_vals.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect();
    m_step(log_vals, &resp, n_zeros, 0.0)
}

/// M-step for tied-variance Gaussian mixture with analytic zeros.
fn m_step(log_vals: &[f64], responsibilities: &[f64], n_zeros: usize, r_zero: f64) -> FitParams {
    let n_total = log_vals.len() as f64 + n_zeros as f64;

    let sum_r_nz: f64 = responsibilities.iter().sum();
    let sum_r = sum_r_nz + n_zeros as f64 * r_zero;
    let pi = clamp_probability(sum_r / n_total);

    // Signal mean: zeros contribute 0.0 * r_zero = 0 to the numerator.
    let mu_signal_num: f64 = log_vals.iter().zip(responsibilities).map(|(&x, &r)| r * x).sum();
    let mu_signal = if sum_r > 1e-6 {
        mu_signal_num / sum_r
    } else {
        log_vals.iter().copied().fold(0.0_f64, f64::max)
    };

    // Background mean: zeros contribute 0.0 * (1 - r_zero) = 0 to the numerator.
    let sum_r_bg = n_total - sum_r;
    let mu_bg_num: f64 = log_vals.iter().zip(responsibilities).map(|(&x, &r)| (1.0 - r) * x).sum();
    let mu_bg = if sum_r_bg > 1e-6 { mu_bg_num / sum_r_bg } else { 0.0 };

    // Tied variance: pooled across both components over all observations.
    let var_num: f64 = log_vals
        .iter()
        .zip(responsibilities)
        .map(|(&x, &r)| r * (x - mu_signal).powi(2) + (1.0 - r) * (x - mu_bg).powi(2))
        .sum::<f64>()
        + n_zeros as f64
            * (r_zero * mu_signal.powi(2) + (1.0 - r_zero) * mu_bg.powi(2));
    let sigma = (var_num / n_total.max(1.0)).sqrt().max(0.1);

    FitParams { pi, mu_bg, mu_signal, sigma }
}

fn posterior_signal(x: f64, params: FitParams) -> f64 {
    let log_bg = (1.0 - params.pi).ln() + log_normal_pdf(x, params.mu_bg, params.sigma);
    let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma);
    let denom = logsumexp2(log_bg, log_sig);
    (log_sig - denom).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, Covariates, GuideMetadata};
    use arrow::array::{BooleanArray, Float32Array, StringBuilder};

    fn make_input(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        cell_names: &[&str],
        guide_names: &[&str],
    ) -> crate::data::LoadedInput {
        assert_eq!(cell_names.len(), n_cells);
        assert_eq!(guide_names.len(), n_guides);

        let mut sorted = triples;
        sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let nnz = sorted.len();
        let mut row_offsets = vec![0usize; n_cells + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut last = 0usize;
        for (idx, &(r, c, v)) in sorted.iter().enumerate() {
            while last < r {
                row_offsets[last + 1] = idx;
                last += 1;
            }
            col_indices.push(c);
            values.push(v);
        }
        for i in (last + 1)..=n_cells {
            row_offsets[i] = nnz;
        }

        let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values).unwrap();
        let mut bc = StringBuilder::new();
        cell_names.iter().for_each(|s| bc.append_value(s));
        let mut gd = StringBuilder::new();
        guide_names.iter().for_each(|s| gd.append_value(s));

        crate::data::LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(vec![0.0f32; n_cells]),
                batch: crate::data::BatchLabels::single_batch(n_cells),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn is_unassigned(r: &crate::data::AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    // ---------------------------------------------------------------------------
    // Assignment behaviour
    // ---------------------------------------------------------------------------

    #[test]
    fn assigns_high_signal_cells() {
        // 6 cells × 1 guide. Cells 0/2 have zero counts (absent from sparse matrix).
        // Cell 1 has 1 UMI — noise level. Cells 3/4/5 have 25/30/22 — clear signal.
        // On log10 scale: 0.0 vs ~1.4. GMM should separate them.
        let input = make_input(
            6, 1,
            vec![(1, 0, 1), (3, 0, 25), (4, 0, 30), (5, 0, 22)],
            &["C1", "C2", "C3", "C4", "C5", "C6"],
            &["gA"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);

        assert!(is_u.value(0), "C1 (0 counts) should be unassigned");
        assert!(is_u.value(1), "C2 (1 count) should be unassigned");
        assert!(is_u.value(2), "C3 (0 counts) should be unassigned");
        assert!(!is_u.value(3), "C4 (25) should be assigned");
        assert!(!is_u.value(4), "C5 (30) should be assigned");
        assert!(!is_u.value(5), "C6 (22) should be assigned");
    }

    #[test]
    fn skips_guides_with_too_little_signal() {
        // max_count = 1 < min_max_count = 2 → fit_guide returns None.
        // No cell can pass a threshold that doesn't exist.
        let input = make_input(
            3, 1,
            vec![(1, 0, 1), (2, 0, 1)],
            &["C1", "C2", "C3"],
            &["gA"],
        );
        let result = GaussModel::default().assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        assert!(is_u.value(0));
        assert!(is_u.value(1));
        assert!(is_u.value(2));
    }

    #[test]
    fn guide_name_in_output_correct() {
        // Dictionary-encoded guide_id must resolve to the correct guide name.
        let input = make_input(
            4, 1,
            vec![(1, 0, 1), (2, 0, 25), (3, 0, 30)],
            &["C1", "C2", "C3", "C4"],
            &["sgTP53_1"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        let assigned_cells: Vec<usize> = (0..4).filter(|&i| !is_u.value(i)).collect();
        assert!(!assigned_cells.is_empty());

        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int16Type;
        let dict = result.batch.column_by_name("guide_id").unwrap()
            .as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
        let values = dict.values().as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        for &i in &assigned_cells {
            let key = dict.keys().value(i) as usize;
            assert_eq!(values.value(key), "sgTP53_1");
        }
    }

    #[test]
    fn multi_infected_flagged() {
        // C4 (index 3) has signal-level counts for both gA (25) and gB (28) → multi-infected.
        // Both guides need enough noise + signal observations for the GMM to converge cleanly.
        let input = make_input(
            8, 2,
            vec![
                (0, 0, 1), (0, 1, 1),
                (1, 0, 1), (1, 1, 1),
                (2, 0, 1),
                (3, 0, 25), (3, 1, 28),
                (4, 0, 26),
                (5, 0, 24),
                (6, 1, 29),
                (7, 1, 30),
            ],
            &["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            &["gA", "gB"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(3), "C4 should be multi-infected");
        assert!(!is_multi.value(0), "C1 is noise only, not multi-infected");
    }

    #[test]
    fn n_guides_detected_matches_n_passing() {
        // gA: signal in C3 (25) and C4 (26). gB: signal in C3 (28) and C5 (30).
        // C3 passes both → n_guides_detected = 2. C4 passes only gA → 1.
        let input = make_input(
            5, 2,
            vec![(1, 0, 1), (2, 0, 25), (2, 1, 28), (3, 0, 26), (4, 1, 30)],
            &["C1", "C2", "C3", "C4", "C5"],
            &["gA", "gB"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(n_det.value(2), 2, "C3 detected 2 guides");
        assert_eq!(n_det.value(3), 1, "C4 detected 1 guide");
    }

    #[test]
    fn assigned_x_matches_batch() {
        // Assigned cells must have exactly one nonzero in assigned_x; unassigned cells none.
        let input = make_input(
            4, 1,
            vec![(1, 0, 1), (2, 0, 25), (3, 0, 30)],
            &["C1", "C2", "C3", "C4"],
            &["gA"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = is_unassigned(&result);
        for cell in 0..4 {
            let row = result.assigned_x.get_row(cell).unwrap();
            if !is_u.value(cell) {
                assert_eq!(row.nnz(), 1, "cell {cell} should have 1 assigned guide");
                assert_eq!(row.values(), &[1u8]);
            } else {
                assert_eq!(row.nnz(), 0, "cell {cell} should have no assigned guide");
            }
        }
    }

    #[test]
    fn cell_barcodes_in_output_come_from_input() {
        // Output row i always corresponds to input cell i; barcodes are never reordered.
        let input = make_input(
            3, 1,
            vec![(1, 0, 25), (2, 0, 30)],
            &["BARCODE_A", "BARCODE_B", "BARCODE_C"],
            &["gX"],
        );
        let model = GaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let bc = result.batch.column_by_name("cell_barcode").unwrap()
            .as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        assert_eq!(bc.value(0), "BARCODE_A");
        assert_eq!(bc.value(1), "BARCODE_B");
        assert_eq!(bc.value(2), "BARCODE_C");
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![], &[], &[]);
        let result = GaussModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.assigned_x.nnz(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        // All counts = 1 → max = 1 < min_max_count = 2; no cell assigned.
        use arrow::array::Array;
        let input = make_input(
            3, 1,
            vec![(1, 0, 1), (2, 0, 1)],
            &["C1", "C2", "C3"],
            &["gA"],
        );
        let result = GaussModel::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("guide_id").unwrap().is_null(i));
            assert!(result.batch.column_by_name("umi_count").unwrap().is_null(i));
            assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(i));
        }
    }

    // ---------------------------------------------------------------------------
    // EM math unit tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fit_mixture_separates_bimodal_data() {
        // Background cluster near log10(2) ≈ 0.30; signal cluster near log10(26) ≈ 1.41.
        // After convergence, mu_signal must exceed mu_bg.
        let bg: Vec<f64> = (0..15).map(|_| 0.301).collect(); // log10(2)
        let sig: Vec<f64> = vec![1.408, 1.431, 1.415, 1.447, 1.380]; // log10(25..30)
        let all: Vec<f64> = bg.into_iter().chain(sig).collect();
        let params = fit_mixture(&all, 100, 100, 1e-8);
        assert!(
            params.mu_signal > params.mu_bg,
            "mu_signal ({}) should exceed mu_bg ({})",
            params.mu_signal, params.mu_bg
        );
    }

    #[test]
    fn posterior_signal_high_for_large_count() {
        // Fit on a clearly bimodal set, then evaluate posteriors at known-background
        // and known-signal log10 values. Signal posterior must be high at count 25,
        // low at count 1.
        let nonzero: Vec<f64> = vec![0.301, 0.322, 0.342, 1.408, 1.431, 1.447, 1.462];
        let params = fit_mixture(&nonzero, 50, 100, 1e-8);
        let p_high = posterior_signal((25.0_f64 + 1.0).log10(), params);
        let p_low = posterior_signal((1.0_f64 + 1.0).log10(), params);
        assert!(p_high > 0.9, "posterior at signal level should be > 0.9, got {p_high}");
        assert!(p_low < 0.1, "posterior at background level should be < 0.1, got {p_low}");
    }

    #[test]
    fn fit_guide_returns_none_for_tiny_data() {
        // Only 1 nonzero value → below min_nonzero = 2.
        let model = GaussModel::default();
        assert_eq!(model.fit_guide(&[0, 5, 0], 10), None);
    }

    #[test]
    fn fit_guide_returns_none_for_low_max() {
        // max_count = 1 < min_max_count = 2.
        let model = GaussModel::default();
        assert_eq!(model.fit_guide(&[0, 1, 1, 1], 10), None);
    }

    #[test]
    fn zeros_included_in_background_estimate() {
        // More zeros should pull mu_bg toward 0 (log10(1) = 0.0) and lower sigma.
        let nonzero: Vec<f64> = vec![0.301, 0.322, 1.408, 1.431, 1.447];
        let params_few = fit_mixture(&nonzero, 5, 100, 1e-8);
        let params_many = fit_mixture(&nonzero, 1000, 100, 1e-8);
        assert!(
            params_many.mu_bg <= params_few.mu_bg + 0.05,
            "more zeros should pull mu_bg toward 0"
        );
    }

    #[test]
    fn gauss_posterior_equals_pi_when_components_identical() {
        // μ_signal = μ_bg ⇒ Gaussian densities equal at every x ⇒ posterior = π.
        for &pi in &[0.05, 0.3, 0.5, 0.8, 0.95] {
            let p = FitParams { pi, mu_bg: 0.5, mu_signal: 0.5, sigma: 0.3 };
            let post = posterior_signal(1.234, p);
            assert!((post - pi).abs() < 1e-12, "π={pi}: got {post}");
        }
    }
}

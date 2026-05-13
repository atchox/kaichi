use super::AssignmentModel;
use crate::data::{AssignmentResult, LoadedInput};
use crate::schema::assignment_schema_ref;

use anyhow::{Context, Result};
use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, Float32Builder, StringArray,
        StringDictionaryBuilder, UInt32Builder, UInt8Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::*;
use serde_json::{json, Value};
use statrs::function::gamma::ln_gamma;
use std::{f64::consts::PI, sync::Arc};

/// Per-guide Poisson-Gaussian mixture model.
///
/// Background counts are modeled as a continuous Poisson on log2(UMI). Signal
/// counts are modeled as Gaussian on log2(UMI). Each guide is fit independently
/// on non-zero UMI counts via closed-form EM on the per-guide CSC column.
pub struct PoissonGaussModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
}

impl Default for PoissonGaussModel {
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
    lambda_bg: f64,
    mu_signal: f64,
    sigma_signal: f64,
}

impl AssignmentModel for PoissonGaussModel {
    fn name(&self) -> &'static str {
        "poisson_gauss"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();

        // Step 1: per-guide EM in parallel via CSC columns.
        //
        // CSC gives us each guide's nonzero (cell_idx, count) pairs in O(nnz_guide).
        // We fit once per guide and store the result. Step 2 never touches CSC again.
        let guide_fits: Vec<Option<(u32, FitParams)>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                self.fit_guide(col.values())
            })
            .collect();

        // Step 2: iterate cells in CSR row order, applying stored FitParams.
        //
        // CSR gives us each cell's nonzero (guide_idx, count) pairs in O(nnz_cell).
        // For each passing count we evaluate `posterior_signal` using the guide's
        // precomputed params — no CSC access here.
        let csr = input.counts.csr();
        let guide_ids_arr = &input.guide_metadata.guide_ids;
        let cell_barcodes = &input.covariates.cell_barcodes;

        let mut out_guide_ids: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        let mut out_target_genes: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        let mut out_umi_counts = UInt32Builder::with_capacity(n_cells);
        let mut out_confidence = Float32Builder::with_capacity(n_cells);
        let mut out_is_unassigned = BooleanBuilder::with_capacity(n_cells);
        let mut out_is_multi = BooleanBuilder::with_capacity(n_cells);
        let mut out_n_detected = UInt8Builder::with_capacity(n_cells);

        let mut assigned_triples: Vec<(usize, usize)> = Vec::new();

        for cell in 0..n_cells {
            let row = csr.get_row(cell).unwrap();

            let mut best_guide: Option<(&str, f32, u32)> = None;
            let mut n_passing: usize = 0;

            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values().iter()) {
                // guide_fits[guide_idx] = None means skip (too few data points to fit)
                let Some((threshold, params)) = guide_fits[guide_idx] else { continue };
                if count < threshold { continue }

                let posterior = posterior_signal(log2_count(count), params) as f32;
                if posterior >= self.min_confidence {
                    n_passing += 1;
                    let guide_name = guide_ids_arr.value(guide_idx);
                    match best_guide {
                        None => best_guide = Some((guide_name, posterior, count)),
                        Some((_, best_post, best_count)) => {
                            if posterior > best_post || (posterior == best_post && count > best_count) {
                                best_guide = Some((guide_name, posterior, count));
                            }
                        }
                    }
                    assigned_triples.push((cell, guide_idx));
                }
            }

            out_n_detected.append_value(n_passing.min(255) as u8);

            match best_guide {
                None => {
                    out_guide_ids.append_null();
                    out_target_genes.append_null();
                    out_umi_counts.append_null();
                    out_confidence.append_null();
                    out_is_unassigned.append_value(true);
                    out_is_multi.append_value(false);
                }
                Some((guide, posterior, count)) => {
                    out_guide_ids.append_value(guide);
                    out_target_genes.append_null();
                    out_umi_counts.append_value(count);
                    out_confidence.append_value(posterior);
                    out_is_unassigned.append_value(false);
                    out_is_multi.append_value(n_passing > 1);
                }
            }
        }

        let model_name_arr: DictionaryArray<arrow::datatypes::Int8Type> = {
            let values = StringArray::from(vec![self.name()]);
            let keys = arrow::array::Int8Array::from(vec![0i8; n_cells]);
            DictionaryArray::try_new(keys, Arc::new(values))
                .context("failed to build assignment_model dictionary")?
        };

        let batch = RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                Arc::new(cell_barcodes.clone()),
                Arc::new(out_guide_ids.finish()),
                Arc::new(out_target_genes.finish()),
                Arc::new(out_umi_counts.finish()),
                Arc::new(out_confidence.finish()),
                Arc::new(model_name_arr),
                Arc::new(out_is_unassigned.finish()),
                Arc::new(out_is_multi.finish()),
                Arc::new(out_n_detected.finish()),
            ],
        )
        .context("failed to build output RecordBatch")?;

        assigned_triples.sort_unstable();
        assigned_triples.dedup();
        let assigned_x = build_assigned_csr(assigned_triples, n_cells, n_guides);

        Ok(AssignmentResult { batch, assigned_x })
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

impl PoissonGaussModel {
    /// Fit EM for one guide and return `(threshold, FitParams)`, or `None` if
    /// the guide has too few observations to fit reliably.
    ///
    /// `counts` is the slice of nonzero values from the guide's CSC column.
    fn fit_guide(&self, counts: &[u32]) -> Option<(u32, FitParams)> {
        let n_nonzero = counts.iter().filter(|&&c| c > 0).count() as u32;
        let max_count = counts.iter().copied().max().unwrap_or(0);

        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }

        let log_counts: Vec<f64> = counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| log2_count(c))
            .collect();

        let params = fit_transformed_mixture(&log_counts, self.max_em_iters, self.tol as f64);

        let threshold = (1..=max_count)
            .find(|&c| posterior_signal(log2_count(c), params) > self.min_confidence as f64)?;

        Some((threshold, params))
    }
}

fn build_assigned_csr(
    sorted_triples: Vec<(usize, usize)>,
    n_cells: usize,
    n_guides: usize,
) -> CsrMatrix<u8> {
    let nnz = sorted_triples.len();
    let mut row_offsets = vec![0usize; n_cells + 1];
    let mut col_indices = Vec::with_capacity(nnz);
    let values = vec![1u8; nnz];

    let mut last_row = 0usize;
    for (idx, &(r, c)) in sorted_triples.iter().enumerate() {
        while last_row < r {
            row_offsets[last_row + 1] = idx;
            last_row += 1;
        }
        col_indices.push(c);
    }
    for i in (last_row + 1)..=n_cells {
        row_offsets[i] = nnz;
    }

    CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, values)
        .expect("assigned CSR construction cannot fail: triples are sorted and deduplicated")
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn fit_transformed_mixture(data: &[f64], max_em_iters: u32, tol: f64) -> FitParams {
    let mut params = initialize_params(data);
    let mut last_log_lik = f64::NEG_INFINITY;
    let mut responsibilities = vec![0.0f64; data.len()];

    for _ in 0..max_em_iters {
        let mut log_lik = 0.0;
        for (idx, &value) in data.iter().enumerate() {
            let log_bg = (1.0 - params.pi).ln()
                + log_continuous_poisson_pmf(value, params.lambda_bg);
            let log_signal = params.pi.ln()
                + log_normal_pdf(value, params.mu_signal, params.sigma_signal);
            let denom = logsumexp2(log_bg, log_signal);
            responsibilities[idx] = (log_signal - denom).exp();
            log_lik += denom;
        }
        if last_log_lik.is_finite() {
            let scale = last_log_lik.abs().max(1.0);
            if (log_lik - last_log_lik).abs() / scale < tol {
                break;
            }
        }
        last_log_lik = log_lik;
        params = m_step(data, &responsibilities);
    }
    params
}

fn posterior_signal(value: f64, params: FitParams) -> f64 {
    let log_bg = (1.0 - params.pi).ln()
        + log_continuous_poisson_pmf(value, params.lambda_bg);
    let log_signal = params.pi.ln()
        + log_normal_pdf(value, params.mu_signal, params.sigma_signal);
    let denom = logsumexp2(log_bg, log_signal);
    (log_signal - denom).exp()
}

fn log2_count(count: u32) -> f64 {
    (count as f64).log2()
}

fn initialize_params(data: &[f64]) -> FitParams {
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    let threshold_idx = ((sorted.len() as f64) * 0.9).floor() as usize;
    let threshold = sorted[threshold_idx.min(sorted.len() - 1)];
    let resp: Vec<f64> = data.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect();
    m_step(data, &resp)
}

fn m_step(data: &[f64], responsibilities: &[f64]) -> FitParams {
    let n = data.len() as f64;
    let sum_r: f64 = responsibilities.iter().sum();
    let sum_bg = n - sum_r;

    let pi = clamp_probability(sum_r / n);

    let lambda_num: f64 = data.iter().zip(responsibilities)
        .map(|(&v, &r)| (1.0 - r) * v).sum();
    let lambda_bg = (lambda_num / sum_bg.max(1e-6)).max(1e-6);

    let mu_num: f64 = data.iter().zip(responsibilities)
        .map(|(&v, &r)| r * v).sum();
    let mu_signal = if sum_r > 1e-6 {
        mu_num / sum_r
    } else {
        data.iter().copied().fold(0.0_f64, f64::max)
    };

    let var_num: f64 = data.iter().zip(responsibilities)
        .map(|(&v, &r)| {
            let c = v - mu_signal;
            r * c * c
        })
        .sum();
    let sigma_signal = (var_num / sum_r.max(1e-6)).sqrt().max(0.25);

    FitParams { pi, lambda_bg, mu_signal, sigma_signal }
}

fn clamp_probability(v: f64) -> f64 {
    v.clamp(1e-6, 1.0 - 1e-6)
}

fn log_continuous_poisson_pmf(value: f64, lambda: f64) -> f64 {
    value * lambda.ln() - lambda - ln_gamma(value + 1.0)
}

fn log_normal_pdf(value: f64, mean: f64, sigma: f64) -> f64 {
    let z = (value - mean) / sigma;
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * z * z
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{CountMatrix, Covariates, GuideMetadata};
    use arrow::array::{BooleanArray, Float32Array, StringBuilder};
    use std::sync::Arc;

    /// Build a `LoadedInput` from (cell_idx, guide_idx, count) triples.
    fn make_input(
        n_cells: usize,
        n_guides: usize,
        triples: Vec<(usize, usize, u32)>,
        cell_names: &[&str],
        guide_names: &[&str],
    ) -> LoadedInput {
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

        LoadedInput {
            counts,
            covariates: Covariates {
                cell_barcodes: bc.finish(),
                total_counts: Float32Array::from(vec![0.0f32; n_cells]),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn get_is_unassigned(result: &AssignmentResult) -> &BooleanArray {
        result.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    // ---------------------------------------------------------------------------
    // Functional tests
    // ---------------------------------------------------------------------------

    #[test]
    fn assigns_high_signal_cells() {
        // 6 cells × 1 guide: C4/C5/C6 have strong signal
        let input = make_input(
            6, 1,
            vec![(1, 0, 1), (3, 0, 25), (4, 0, 30), (5, 0, 22)],
            &["C1", "C2", "C3", "C4", "C5", "C6"],
            &["gA"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = get_is_unassigned(&result);

        assert!(is_u.value(0), "C1 (0 counts) should be unassigned");
        assert!(is_u.value(1), "C2 (1 count) should be unassigned");
        assert!(is_u.value(2), "C3 (0 counts) should be unassigned");
        assert!(!is_u.value(3), "C4 (25) should be assigned");
        assert!(!is_u.value(4), "C5 (30) should be assigned");
        assert!(!is_u.value(5), "C6 (22) should be assigned");
    }

    #[test]
    fn skips_guides_with_too_little_signal() {
        // All counts <= 1 → model never finds a threshold
        let input = make_input(
            3, 1,
            vec![(1, 0, 1), (2, 0, 1)],
            &["C1", "C2", "C3"],
            &["gA"],
        );
        let result = PoissonGaussModel::default().assign(&input).unwrap();
        let is_u = get_is_unassigned(&result);
        assert!(is_u.value(0));
        assert!(is_u.value(1));
        assert!(is_u.value(2));
    }

    #[test]
    fn guide_name_in_output_correct() {
        let input = make_input(
            4, 1,
            vec![(1, 0, 1), (2, 0, 25), (3, 0, 30)],
            &["C1", "C2", "C3", "C4"],
            &["sgTP53_1"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = get_is_unassigned(&result);

        // Find an assigned cell
        let assigned_cells: Vec<usize> = (0..4).filter(|&i| !is_u.value(i)).collect();
        assert!(!assigned_cells.is_empty(), "at least one cell should be assigned");

        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int16Type;
        let guide_col = result.batch.column_by_name("guide_id").unwrap();
        let dict = guide_col.as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
        let values = dict.values().as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
        for &i in &assigned_cells {
            let key = dict.keys().value(i) as usize;
            assert_eq!(values.value(key), "sgTP53_1");
        }
    }

    #[test]
    fn multi_infected_flagged() {
        // 2 guides, both signal-level in the same cell
        let input = make_input(
            5, 2,
            vec![
                (0, 0, 1), (0, 1, 1),
                (1, 0, 25), (1, 1, 28),  // multi-infected
                (2, 0, 26),
                (3, 0, 24),
                (4, 1, 29),
            ],
            &["C1", "C2", "C3", "C4", "C5"],
            &["gA", "gB"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();

        use arrow::array::BooleanArray;
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        // C2 (idx 1) has both guides at signal level
        assert!(is_multi.value(1), "C2 should be multi-infected");
        // C1 has only noise counts
        assert!(!is_multi.value(0));
    }

    #[test]
    fn n_guides_detected_matches_n_passing() {
        // Need ≥2 nonzeros per guide for fit_guide to proceed (min_nonzero=2).
        // gA: C3=25, C4=26  gB: C3=28, C5=30  → C3 detects 2 guides
        let input = make_input(
            5, 2,
            vec![(1, 0, 1), (2, 0, 25), (2, 1, 28), (3, 0, 26), (4, 1, 30)],
            &["C1", "C2", "C3", "C4", "C5"],
            &["gA", "gB"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();

        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();

        // C3 should have n_guides_detected = 2 (both gA and gB passing)
        assert_eq!(n_det.value(2), 2, "C3 detected 2 guides");
        // C4 should have n_guides_detected = 1
        assert_eq!(n_det.value(3), 1, "C4 detected 1 guide");
    }

    #[test]
    fn assigned_x_matches_batch() {
        let input = make_input(
            4, 1,
            vec![(1, 0, 1), (2, 0, 25), (3, 0, 30)],
            &["C1", "C2", "C3", "C4"],
            &["gA"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();
        let is_u = get_is_unassigned(&result);

        // assigned_x should have a 1 exactly where is_unassigned = false
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
        let input = make_input(
            3, 1,
            vec![(1, 0, 25), (2, 0, 30)],
            &["BARCODE_A", "BARCODE_B", "BARCODE_C"],
            &["gX"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
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
        let result = PoissonGaussModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.assigned_x.nnz(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        // All noise counts — guide never gets enough signal to build a threshold
        let input = make_input(
            3, 1,
            vec![(1, 0, 1), (2, 0, 1)],
            &["C1", "C2", "C3"],
            &["gA"],
        );
        let result = PoissonGaussModel::default().assign(&input).unwrap();
        for i in 0..3 {
            assert!(result.batch.column_by_name("is_unassigned").unwrap()
                .as_any().downcast_ref::<BooleanArray>().unwrap().value(i));
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
        // Low cluster: 0.2..0.8, high cluster: 4.0..5.0
        let data: Vec<f64> = (0..20)
            .map(|i| if i < 15 { 0.5 + i as f64 * 0.02 } else { 4.0 + (i - 15) as f64 * 0.25 })
            .collect();
        let params = fit_transformed_mixture(&data, 100, 1e-8);
        // Signal component should have mu > background lambda
        assert!(
            params.mu_signal > params.lambda_bg,
            "mu_signal ({}) should exceed lambda_bg ({})",
            params.mu_signal,
            params.lambda_bg
        );
    }

    #[test]
    fn posterior_signal_high_for_large_count() {
        let data: Vec<f64> = vec![0.3, 0.4, 0.5, 4.5, 4.8, 5.0, 5.2];
        let params = fit_transformed_mixture(&data, 100, 1e-8);
        let p_high = posterior_signal(5.0, params);
        let p_low = posterior_signal(0.3, params);
        assert!(p_high > 0.9, "posterior at signal level should be > 0.9, got {p_high}");
        assert!(p_low < 0.1, "posterior at background level should be < 0.1, got {p_low}");
    }

    #[test]
    fn logsumexp2_numerically_stable() {
        // Large difference: should not underflow
        let r = logsumexp2(1000.0, -1000.0);
        assert!(r.is_finite(), "logsumexp2 overflowed");
        assert!((r - 1000.0).abs() < 1.0, "logsumexp2 dominated by large term");
    }

    #[test]
    fn fit_guide_returns_none_for_tiny_data() {
        let model = PoissonGaussModel::default();
        // Only 1 nonzero — below min_nonzero = 2
        assert_eq!(model.fit_guide(&[0, 5, 0]), None);
    }

    #[test]
    fn fit_guide_returns_none_for_low_max() {
        let model = PoissonGaussModel::default();
        // max_count = 1 < min_max_count = 2
        assert_eq!(model.fit_guide(&[0, 1, 1, 1]), None);
    }
}

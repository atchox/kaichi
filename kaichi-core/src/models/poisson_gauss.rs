use super::AssignmentModel;
use super::em::{clamp_probability, log_normal_pdf, log_poisson_pmf, logsumexp2, run_em, run_em_multistart};
use super::em_count_mixture::decide_threshold;
use super::TwoStage;
use crate::data::{AssignmentResult, LoadedInput};
use crate::score::{ModelKind, ScoreMatrix};

use anyhow::Result;
use arrow::array::{Float32Array, UInt32Array};
use rayon::prelude::*;
use serde_json::{json, Value};

/// Per-guide Poisson-Gaussian mixture model on raw UMI counts.
///
/// Background counts ~ Poisson(λ) on raw UMIs. Signal counts ~ N(μ, σ²) on
/// raw UMIs. Each guide is fit independently via closed-form EM over all cells,
/// including zeros (which are valid Poisson background observations). An integer
/// UMI threshold is derived as the first count where the signal posterior exceeds
/// `min_confidence`.
pub struct PoissonGaussModel {
    pub min_confidence: f32,
    pub max_em_iters: u32,
    pub tol: f32,
    pub min_nonzero: u32,
    pub min_max_count: u32,
    pub n_restarts: u32,
}

impl Default for PoissonGaussModel {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_em_iters: 100,
            tol: 1e-6,
            min_nonzero: 2,
            min_max_count: 2,
            n_restarts: 5,
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

// ---------------------------------------------------------------------------
// AssignmentModel
// ---------------------------------------------------------------------------

impl AssignmentModel for PoissonGaussModel {
    fn name(&self) -> &'static str {
        "poisson_gauss"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let scores = self.score(input)?;
        self.decide(&scores, self.min_confidence)
    }

    fn params_json(&self) -> Value {
        json!({
            "min_confidence": self.min_confidence,
            "max_em_iters": self.max_em_iters,
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

impl TwoStage for PoissonGaussModel {
    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csc = input.counts.csc();

        // Per-guide EM in parallel via CSC columns.
        // Returns (threshold, params) or None if gated out.
        let guide_fits: Vec<Option<(u32, FitParams)>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let n_zeros = n_cells - col.nnz();
                self.fit_guide(col.values(), n_zeros)
            })
            .collect();

        // Serial per-cell scoring pass over CSR rows.
        let csr = input.counts.csr();
        let mut score_data: Vec<f32> = Vec::with_capacity(csr.nnz());
        let mut umi_data: Vec<u32> = Vec::with_capacity(csr.nnz());
        let mut score_indices: Vec<u32> = Vec::with_capacity(csr.nnz());
        let mut score_indptr: Vec<u32> = Vec::with_capacity(n_cells + 1);
        score_indptr.push(0);

        for cell in 0..n_cells {
            let row = csr.get_row(cell).unwrap();
            for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
                let Some((threshold, params)) = guide_fits[guide_idx] else { continue };
                // Score is 0.0 for sub-threshold counts; decide() filters these out.
                let score = if count >= threshold {
                    posterior_signal(count as f64, params) as f32
                } else {
                    0.0
                };
                score_data.push(score);
                umi_data.push(count);
                score_indices.push(guide_idx as u32);
            }
            score_indptr.push(score_data.len() as u32);
        }

        Ok(ScoreMatrix {
            data: Float32Array::from(score_data),
            umi_counts: UInt32Array::from(umi_data),
            indices: UInt32Array::from(score_indices),
            indptr: UInt32Array::from(score_indptr),
            cell_barcodes: input.covariates.cell_barcodes.clone(),
            guide_ids: input.guide_metadata.guide_ids.clone(),
            model: ModelKind::PoissonGauss,
            model_params: self.params_json(),
        })
    }

    fn decide(&self, scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult> {
        decide_threshold(scores, min_confidence)
    }
}

// ---------------------------------------------------------------------------
// Fitting
// ---------------------------------------------------------------------------

impl PoissonGaussModel {
    fn fit_guide(&self, nonzero_counts: &[u32], n_zeros: usize) -> Option<(u32, FitParams)> {
        let n_nonzero = nonzero_counts.iter().filter(|&&c| c > 0).count() as u32;
        let max_count = nonzero_counts.iter().copied().max().unwrap_or(0);

        if n_nonzero < self.min_nonzero || max_count < self.min_max_count {
            return None;
        }

        let raw_counts: Vec<f64> = nonzero_counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64)
            .collect();

        let params = fit_mixture(&raw_counts, n_zeros, self.max_em_iters, self.tol as f64, self.n_restarts);

        let threshold = (1..=max_count)
            .find(|&c| posterior_signal(c as f64, params) > self.min_confidence as f64)?;

        Some((threshold, params))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

fn fit_mixture(nonzero_vals: &[f64], n_zeros: usize, max_em_iters: u32, tol: f64, n_restarts: u32) -> FitParams {
    let base_init = initialize_params(nonzero_vals, n_zeros);
    run_em_multistart(n_restarts, |pi_k| {
        let mut init = base_init;
        init.pi = pi_k;
        let mut responsibilities = vec![0.0f64; nonzero_vals.len()];
        run_em(
            init,
            |params| {
                let mut ll = 0.0;
                for (idx, &x) in nonzero_vals.iter().enumerate() {
                    let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(x, params.lambda_bg);
                    let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma_signal);
                    let denom = logsumexp2(log_bg, log_sig);
                    responsibilities[idx] = (log_sig - denom).exp();
                    ll += denom;
                }
                let log_bg_zero = (1.0 - params.pi).ln() + log_poisson_pmf(0.0, params.lambda_bg);
                let log_sig_zero = params.pi.ln() + log_normal_pdf(0.0, params.mu_signal, params.sigma_signal);
                let denom_zero = logsumexp2(log_bg_zero, log_sig_zero);
                let r_zero = (log_sig_zero - denom_zero).exp();
                ll += n_zeros as f64 * denom_zero;
                let new_params = m_step(nonzero_vals, &responsibilities, n_zeros, r_zero);
                (new_params, ll)
            },
            max_em_iters,
            tol,
        )
    })
}

fn initialize_params(nonzero_vals: &[f64], n_zeros: usize) -> FitParams {
    let mut sorted = nonzero_vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let threshold_idx = ((sorted.len() as f64) * 0.9).floor() as usize;
    let threshold = sorted[threshold_idx.min(sorted.len() - 1)];
    let resp: Vec<f64> = nonzero_vals.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect();
    m_step(nonzero_vals, &resp, n_zeros, 0.0)
}

fn m_step(nonzero_vals: &[f64], responsibilities: &[f64], n_zeros: usize, r_zero: f64) -> FitParams {
    let sum_r_nonzero: f64 = responsibilities.iter().sum();
    let sum_r_zeros = n_zeros as f64 * r_zero;
    let sum_r = sum_r_nonzero + sum_r_zeros;
    let n_total = nonzero_vals.len() as f64 + n_zeros as f64;

    let pi = clamp_probability(sum_r / n_total);

    let lambda_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| (1.0 - r) * x).sum();
    let lambda_denom = n_total - sum_r;
    let lambda_bg = (lambda_num / lambda_denom.max(1e-6)).max(1e-6);

    let mu_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| r * x).sum::<f64>();
    let mu_signal = if sum_r > 1e-6 {
        mu_num / sum_r
    } else {
        nonzero_vals.iter().copied().fold(0.0_f64, f64::max)
    };

    let var_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| r * (x - mu_signal).powi(2)).sum::<f64>()
        + n_zeros as f64 * r_zero * mu_signal.powi(2);
    let sigma_signal = (var_num / sum_r.max(1e-6)).sqrt().max(1.0);

    FitParams { pi, lambda_bg, mu_signal, sigma_signal }
}

fn posterior_signal(x: f64, params: FitParams) -> f64 {
    let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(x, params.lambda_bg);
    let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma_signal);
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
                batch: crate::data::BatchLabels::single_batch(n_cells),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn get_is_unassigned(result: &AssignmentResult) -> &BooleanArray {
        result.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    #[test]
    fn assigns_high_signal_cells() {
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
        let input = make_input(
            5, 2,
            vec![
                (0, 0, 1), (0, 1, 1),
                (1, 0, 25), (1, 1, 28),
                (2, 0, 26),
                (3, 0, 24),
                (4, 1, 29),
            ],
            &["C1", "C2", "C3", "C4", "C5"],
            &["gA", "gB"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let result = model.assign(&input).unwrap();

        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(1), "C2 should be multi-infected");
        assert!(!is_multi.value(0), "C1 has only noise counts, not multi-infected");
    }

    #[test]
    fn n_guides_detected_matches_n_passing() {
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

        assert_eq!(n_det.value(2), 2, "C3 detected 2 guides");
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

    #[test]
    fn score_and_decide_matches_assign() {
        let input = make_input(
            6, 1,
            vec![(1, 0, 1), (3, 0, 25), (4, 0, 30), (5, 0, 22)],
            &["C1", "C2", "C3", "C4", "C5", "C6"],
            &["gA"],
        );
        let model = PoissonGaussModel { min_confidence: 0.5, ..Default::default() };
        let direct = model.assign(&input).unwrap();
        let scores = model.score(&input).unwrap();
        let via_decide = model.decide(&scores, 0.5).unwrap();

        let is_u_d = get_is_unassigned(&direct);
        let is_u_s = via_decide.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        for i in 0..6 {
            assert_eq!(is_u_d.value(i), is_u_s.value(i), "cell {i}");
        }
    }

    // EM math tests (unchanged)

    #[test]
    fn fit_mixture_separates_bimodal_data() {
        let nonzero: Vec<f64> = (0..20)
            .map(|i| if i < 15 { 1.0 + i as f64 * 0.1 } else { 20.0 + (i - 15) as f64 * 2.0 })
            .collect();
        let params = fit_mixture(&nonzero, 100, 100, 1e-8, 5);
        assert!(
            params.mu_signal > params.lambda_bg,
            "mu_signal ({}) should exceed lambda_bg ({})",
            params.mu_signal, params.lambda_bg
        );
    }

    #[test]
    fn posterior_signal_high_for_large_count() {
        let nonzero: Vec<f64> = vec![1.0, 1.5, 2.0, 20.0, 22.0, 25.0, 28.0];
        let params = fit_mixture(&nonzero, 50, 100, 1e-8, 5);
        let p_high = posterior_signal(25.0, params);
        let p_low = posterior_signal(1.0, params);
        assert!(p_high > 0.9, "posterior at signal level should be > 0.9, got {p_high}");
        assert!(p_low < 0.1, "posterior at background level should be < 0.1, got {p_low}");
    }

    #[test]
    fn fit_guide_returns_none_for_tiny_data() {
        let model = PoissonGaussModel::default();
        assert_eq!(model.fit_guide(&[0, 5, 0], 10), None);
    }

    #[test]
    fn fit_guide_returns_none_for_low_max() {
        let model = PoissonGaussModel::default();
        assert_eq!(model.fit_guide(&[0, 1, 1, 1], 10), None);
    }

    #[test]
    fn zeros_included_in_background_estimate() {
        let nonzero: Vec<f64> = vec![1.0, 2.0, 25.0, 28.0, 30.0];
        let params_few_zeros = fit_mixture(&nonzero, 5, 100, 1e-8, 5);
        let params_many_zeros = fit_mixture(&nonzero, 1000, 100, 1e-8, 5);
        assert!(
            params_many_zeros.lambda_bg < params_few_zeros.lambda_bg,
            "more zeros should pull lambda_bg down"
        );
    }
}

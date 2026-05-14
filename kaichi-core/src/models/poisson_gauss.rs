use super::AssignmentModel;
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::{json, Value};
use statrs::function::gamma::ln_gamma;
use std::f64::consts::PI;

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
        let guide_fits: Vec<Option<(u32, FitParams)>> = (0..n_guides)
            .into_par_iter()
            .map(|g| {
                let col = csc.get_col(g).unwrap();
                let n_zeros = n_cells - col.nnz();
                self.fit_guide(col.values(), n_zeros)
            })
            .collect();

        // Step 2: iterate cells in CSR row order, applying stored FitParams.
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
                if count < threshold { continue }

                let posterior = posterior_signal(count as f64, params) as f32;
                if posterior >= self.min_confidence {
                    n_passing += 1;
                    match best {
                        None => best = Some((guide_idx, posterior, count)),
                        Some((_, best_post, best_count)) => {
                            if posterior > best_post || (posterior == best_post && count > best_count) {
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

        // Every passing guide was pushed in CSR row order (guide-index sorted),
        // exactly once per (cell, guide). Triples are already strictly sorted.
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

impl PoissonGaussModel {
    /// Fit EM for one guide and return `(threshold, FitParams)`, or `None` if
    /// the guide has too few observations to fit reliably.
    ///
    /// `nonzero_counts` are the non-zero UMI values from the guide's CSC column.
    /// `n_zeros` is the number of cells with zero counts for this guide; they are
    /// included analytically in the EM updates as Poisson background observations.
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

        let params = fit_mixture(&raw_counts, n_zeros, self.max_em_iters, self.tol as f64);

        let threshold = (1..=max_count)
            .find(|&c| posterior_signal(c as f64, params) > self.min_confidence as f64)?;

        Some((threshold, params))
    }
}

// ---------------------------------------------------------------------------
// EM
// ---------------------------------------------------------------------------

/// EM on a Poisson-Gaussian mixture over raw UMI counts.
///
/// `nonzero_vals`: non-zero UMI counts as f64.
/// `n_zeros`: number of cells with zero counts for this guide, included
///            analytically so zeros don't need to be materialised.
fn fit_mixture(nonzero_vals: &[f64], n_zeros: usize, max_em_iters: u32, tol: f64) -> FitParams {
    let mut params = initialize_params(nonzero_vals, n_zeros);
    let mut last_log_lik = f64::NEG_INFINITY;
    let mut responsibilities = vec![0.0f64; nonzero_vals.len()];

    for _ in 0..max_em_iters {
        // E-step: responsibilities for non-zero cells.
        let mut log_lik = 0.0;
        for (idx, &x) in nonzero_vals.iter().enumerate() {
            let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(x, params.lambda_bg);
            let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma_signal);
            let denom = logsumexp2(log_bg, log_sig);
            responsibilities[idx] = (log_sig - denom).exp();
            log_lik += denom;
        }

        // E-step: responsibility for a single zero cell (all zeros share the same value).
        let log_bg_zero = (1.0 - params.pi).ln() + log_poisson_pmf(0.0, params.lambda_bg);
        let log_sig_zero = params.pi.ln() + log_normal_pdf(0.0, params.mu_signal, params.sigma_signal);
        let denom_zero = logsumexp2(log_bg_zero, log_sig_zero);
        let r_zero = (log_sig_zero - denom_zero).exp();
        log_lik += n_zeros as f64 * denom_zero;

        if last_log_lik.is_finite() {
            let scale = last_log_lik.abs().max(1.0);
            if (log_lik - last_log_lik).abs() / scale < tol {
                break;
            }
        }
        last_log_lik = log_lik;
        params = m_step(nonzero_vals, &responsibilities, n_zeros, r_zero);
    }
    params
}

fn initialize_params(nonzero_vals: &[f64], n_zeros: usize) -> FitParams {
    let mut sorted = nonzero_vals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let threshold_idx = ((sorted.len() as f64) * 0.9).floor() as usize;
    let threshold = sorted[threshold_idx.min(sorted.len() - 1)];
    let resp: Vec<f64> = nonzero_vals.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect();
    m_step(nonzero_vals, &resp, n_zeros, 0.0)
}

/// M-step with zeros included analytically.
///
/// `r_zero` is the signal responsibility for a single zero cell.
fn m_step(nonzero_vals: &[f64], responsibilities: &[f64], n_zeros: usize, r_zero: f64) -> FitParams {
    let sum_r_nonzero: f64 = responsibilities.iter().sum();
    let sum_r_zeros = n_zeros as f64 * r_zero;
    let sum_r = sum_r_nonzero + sum_r_zeros;
    let n_total = nonzero_vals.len() as f64 + n_zeros as f64;

    let pi = clamp_probability(sum_r / n_total);

    // Poisson background MLE: weighted mean of all counts (zeros contribute 0 to numerator).
    let lambda_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| (1.0 - r) * x).sum();
    let lambda_denom = n_total - sum_r;
    let lambda_bg = (lambda_num / lambda_denom.max(1e-6)).max(1e-6);

    // Gaussian signal MLE.
    let mu_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| r * x).sum::<f64>();
    // Zeros contribute r_zero * 0.0 = 0 to mu numerator.
    let mu_signal = if sum_r > 1e-6 {
        mu_num / sum_r
    } else {
        nonzero_vals.iter().copied().fold(0.0_f64, f64::max)
    };

    let var_num: f64 = nonzero_vals.iter().zip(responsibilities)
        .map(|(&x, &r)| r * (x - mu_signal).powi(2)).sum::<f64>()
        + n_zeros as f64 * r_zero * mu_signal.powi(2); // zero cells: (0 - mu)²
    let sigma_signal = (var_num / sum_r.max(1e-6)).sqrt().max(1.0);

    FitParams { pi, lambda_bg, mu_signal, sigma_signal }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn posterior_signal(x: f64, params: FitParams) -> f64 {
    let log_bg = (1.0 - params.pi).ln() + log_poisson_pmf(x, params.lambda_bg);
    let log_sig = params.pi.ln() + log_normal_pdf(x, params.mu_signal, params.sigma_signal);
    let denom = logsumexp2(log_bg, log_sig);
    (log_sig - denom).exp()
}

fn log_poisson_pmf(x: f64, lambda: f64) -> f64 {
    x * lambda.ln() - lambda - ln_gamma(x + 1.0)
}

fn log_normal_pdf(x: f64, mean: f64, sigma: f64) -> f64 {
    let z = (x - mean) / sigma;
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * z * z
}

fn logsumexp2(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn clamp_probability(v: f64) -> f64 {
    v.clamp(1e-6, 1.0 - 1e-6)
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
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn get_is_unassigned(result: &AssignmentResult) -> &BooleanArray {
        result.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    // ---------------------------------------------------------------------------
    // Assignment behaviour
    // ---------------------------------------------------------------------------

    #[test]
    fn assigns_high_signal_cells() {
        // 6 cells × 1 guide. C1 and C3 have zero counts (implicit in sparse matrix).
        // C2 has a single UMI — noise level. C4/C5/C6 have 25/30/22 — clear signal.
        // The model should find a threshold that separates noise from signal.
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
        // max_count = 1, which is below min_max_count = 2. fit_guide returns None,
        // so no cell can ever pass the threshold — all three cells must be unassigned.
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
        // guide_id is stored as a Dictionary<Int16, Utf8>. Verify the dictionary
        // encoding resolves to the correct guide name for each assigned cell.
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
        // C1 has noise counts for both guides (1 UMI each) — not infected.
        // C2 has signal-level counts for both gA (25) and gB (28) — multi-infected.
        // C3/C4 are singly infected via gA; C5 via gB.
        // is_multi_infected should be true only for C2.
        let input = make_input(
            5, 2,
            vec![
                (0, 0, 1), (0, 1, 1),
                (1, 0, 25), (1, 1, 28),  // C2: both guides at signal level
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
        // gA: signal in C3 (25) and C4 (26). gB: signal in C3 (28) and C5 (30).
        // Each guide has ≥ 2 nonzeros so fit_guide proceeds (min_nonzero = 2).
        // C3 passes threshold for both guides → n_guides_detected = 2.
        // C4 passes only gA → n_guides_detected = 1.
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
        // The sparse assignment matrix assigned_x must mirror is_unassigned:
        // assigned cells have exactly one nonzero (value = 1) in their row;
        // unassigned cells have an empty row.
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
        // Barcodes are passed through from LoadedInput unchanged — the model never
        // reorders or drops cells, so output row i always corresponds to input cell i.
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
        // Zero cells and zero guides: valid edge case, must not panic.
        let input = make_input(0, 0, vec![], &[], &[]);
        let result = PoissonGaussModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.assigned_x.nnz(), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        // When a cell is unassigned, guide_id / umi_count / assignment_confidence
        // must be Arrow null — not zero or empty string — to satisfy the schema contract.
        // Here all counts are noise (max = 1 < min_max_count = 2), so no cell is assigned.
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

    // ---------------------------------------------------------------------------
    // EM math unit tests
    // ---------------------------------------------------------------------------

    #[test]
    fn fit_mixture_separates_bimodal_data() {
        // Synthetic counts with a clear two-component structure:
        // background cluster at 1–3 UMIs, signal cluster at 20–30 UMIs.
        // After convergence, mu_signal must exceed lambda_bg.
        let nonzero: Vec<f64> = (0..20)
            .map(|i| if i < 15 { 1.0 + i as f64 * 0.1 } else { 20.0 + (i - 15) as f64 * 2.0 })
            .collect();
        let params = fit_mixture(&nonzero, 100, 100, 1e-8);
        assert!(
            params.mu_signal > params.lambda_bg,
            "mu_signal ({}) should exceed lambda_bg ({})",
            params.mu_signal, params.lambda_bg
        );
    }

    #[test]
    fn posterior_signal_high_for_large_count() {
        // Fit on a clearly bimodal set, then evaluate posteriors at known-background
        // and known-signal counts. The mixture should assign > 0.9 to signal at 25 UMIs
        // and < 0.1 to signal at 1 UMI.
        let nonzero: Vec<f64> = vec![1.0, 1.5, 2.0, 20.0, 22.0, 25.0, 28.0];
        let params = fit_mixture(&nonzero, 50, 100, 1e-8);
        let p_high = posterior_signal(25.0, params);
        let p_low = posterior_signal(1.0, params);
        assert!(p_high > 0.9, "posterior at signal level should be > 0.9, got {p_high}");
        assert!(p_low < 0.1, "posterior at background level should be < 0.1, got {p_low}");
    }

    #[test]
    fn logsumexp2_numerically_stable() {
        // With inputs differing by 2000, naive exp would overflow/underflow.
        // logsumexp2 must return a finite value dominated by the larger term.
        let r = logsumexp2(1000.0, -1000.0);
        assert!(r.is_finite(), "logsumexp2 overflowed");
        assert!((r - 1000.0).abs() < 1.0, "logsumexp2 dominated by large term");
    }

    #[test]
    fn fit_guide_returns_none_for_tiny_data() {
        // Only 1 nonzero value → below min_nonzero = 2. Cannot fit a mixture reliably.
        let model = PoissonGaussModel::default();
        assert_eq!(model.fit_guide(&[0, 5, 0], 10), None);
    }

    #[test]
    fn fit_guide_returns_none_for_low_max() {
        // max_count = 1 < min_max_count = 2. Even if there are enough nonzeros,
        // we can't distinguish signal from background at this count level.
        let model = PoissonGaussModel::default();
        assert_eq!(model.fit_guide(&[0, 1, 1, 1], 10), None);
    }

    #[test]
    fn zeros_included_in_background_estimate() {
        // Zeros are Poisson background observations. The more zeros, the lower the
        // MLE for lambda_bg. A guide with 1000 zero-count cells should yield a much
        // smaller lambda_bg than the same nonzero counts with only 5 zero cells.
        let nonzero: Vec<f64> = vec![1.0, 2.0, 25.0, 28.0, 30.0];
        let params_few_zeros = fit_mixture(&nonzero, 5, 100, 1e-8);
        let params_many_zeros = fit_mixture(&nonzero, 1000, 100, 1e-8);
        assert!(
            params_many_zeros.lambda_bg < params_few_zeros.lambda_bg,
            "more zeros should pull lambda_bg down"
        );
    }
}

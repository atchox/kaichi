use super::{AssignmentInput, AssignmentModel};
use crate::schema::assignment_schema_ref;

use anyhow::{Context, Result};
use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, Float32Builder, StringArray,
        StringDictionaryBuilder, UInt32Array, UInt32Builder, UInt8Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use statrs::function::gamma::ln_gamma;
use std::{collections::HashMap, f64::consts::PI, sync::Arc};

/// Per-guide Poisson-Gaussian mixture model.
///
/// Background counts are modeled as a continuous Poisson on log2(UMI). Signal
/// counts are modeled as Gaussian on log2(UMI). Each guide is fit independently on
/// non-zero UMI counts, then an integer UMI threshold is derived from the first
/// count where posterior signal probability exceeds `min_confidence`.
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

#[derive(Clone, Copy, Debug)]
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

    fn assign(&self, input: &AssignmentInput) -> Result<RecordBatch> {
        let parsed = ParsedInput::from_assignment_input(input)?;
        let n_cells = parsed.cell_order.len();

        let guide_posteriors: Vec<(String, Vec<f32>)> = parsed
            .guide_order
            .par_iter()
            .map(|guide| {
                let counts = &parsed.counts_by_guide[guide];
                let posteriors = self.threshold_posteriors(counts);
                (guide.clone(), posteriors)
            })
            .collect();

        let mut out_barcodes: Vec<&str> = Vec::with_capacity(n_cells);
        let mut out_guide_ids: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut out_target_genes: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut out_umi_counts = UInt32Builder::with_capacity(n_cells);
        let mut out_confidence = Float32Builder::with_capacity(n_cells);
        let mut out_is_unassigned = BooleanBuilder::with_capacity(n_cells);
        let mut out_is_multi = BooleanBuilder::with_capacity(n_cells);
        let mut out_n_detected = UInt8Builder::with_capacity(n_cells);

        for cell_idx in 0..n_cells {
            out_barcodes.push(parsed.cell_order[cell_idx].as_str());

            let mut passing: Vec<(&str, f32, u32)> = Vec::new();
            for (guide, posteriors) in &guide_posteriors {
                let posterior = posteriors[cell_idx];
                if posterior >= self.min_confidence {
                    let count = parsed.counts_by_guide[guide][cell_idx];
                    passing.push((guide.as_str(), posterior, count));
                }
            }

            let n_detected = passing.len().min(255) as u8;
            out_n_detected.append_value(n_detected);

            if passing.is_empty() {
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                continue;
            }

            passing.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.2.cmp(&a.2)));
            let (guide, posterior, count) = passing[0];

            out_guide_ids.append_value(guide);
            // Guide-library enrichment is a separate step; no metadata is available here.
            out_target_genes.append_null();
            out_umi_counts.append_value(count);
            out_confidence.append_value(posterior);
            out_is_unassigned.append_value(false);
            out_is_multi.append_value(passing.len() > 1);
        }

        let model_name_arr: DictionaryArray<arrow::datatypes::Int8Type> = {
            let values = StringArray::from(vec![self.name()]);
            let keys = arrow::array::Int8Array::from(vec![0i8; n_cells]);
            DictionaryArray::try_new(keys, Arc::new(values))
                .context("failed to build assignment_model dictionary")?
        };

        RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                Arc::new(StringArray::from(out_barcodes)),
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
        .context("failed to build output RecordBatch")
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
    fn threshold_posteriors(&self, counts: &[u32]) -> Vec<f32> {
        let nonzero = counts.iter().filter(|&&count| count > 0).count() as u32;
        let max_count = counts.iter().copied().max().unwrap_or(0);
        if nonzero < self.min_nonzero || max_count < self.min_max_count {
            return vec![0.0; counts.len()];
        }

        let data: Vec<f64> = counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| log2_count(count))
            .collect();
        let params = fit_transformed_mixture(&data, self.max_em_iters, self.tol as f64);

        let threshold = (1..=max_count)
            .find(|&count| posterior_signal(log2_count(count), params) > self.min_confidence as f64);

        let Some(threshold) = threshold else {
            return vec![0.0; counts.len()];
        };

        counts
            .iter()
            .map(|&count| {
                if count >= threshold {
                    posterior_signal(log2_count(count), params) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

fn fit_transformed_mixture(data: &[f64], max_em_iters: u32, tol: f64) -> FitParams {
    let mut params = initialize_params(data);
    let mut last_log_lik = f64::NEG_INFINITY;
    let mut responsibilities = vec![0.0; data.len()];

    for _ in 0..max_em_iters {
        let mut log_lik = 0.0;

        for (idx, &value) in data.iter().enumerate() {
            let log_bg = (1.0 - params.pi).ln() + log_continuous_poisson_pmf(value, params.lambda_bg);
            let log_signal =
                params.pi.ln() + log_normal_pdf(value, params.mu_signal, params.sigma_signal);
            let denom = logsumexp2(log_bg, log_signal);
            responsibilities[idx] = (log_signal - denom).exp();
            log_lik += denom;
        }

        if last_log_lik.is_finite() {
            let denom = last_log_lik.abs().max(1.0);
            if ((log_lik - last_log_lik).abs() / denom) < tol {
                break;
            }
        }
        last_log_lik = log_lik;
        params = m_step(data, &responsibilities);
    }

    params
}

fn posterior_signal(value: f64, params: FitParams) -> f64 {
    let log_bg = (1.0 - params.pi).ln() + log_continuous_poisson_pmf(value, params.lambda_bg);
    let log_signal = params.pi.ln() + log_normal_pdf(value, params.mu_signal, params.sigma_signal);
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

    let responsibilities: Vec<f64> = data
        .iter()
        .map(|&value| if value >= threshold { 1.0 } else { 0.0 })
        .collect();

    m_step(data, &responsibilities)
}

fn m_step(data: &[f64], responsibilities: &[f64]) -> FitParams {
    let n = data.len() as f64;
    let sum_r: f64 = responsibilities.iter().sum();
    let sum_bg = n - sum_r;

    let pi = clamp_probability(sum_r / n);

    let lambda_num: f64 = data
        .iter()
        .zip(responsibilities)
        .map(|(&value, &r)| (1.0 - r) * value)
        .sum();
    let lambda_bg = (lambda_num / sum_bg.max(1e-6)).max(1e-6);

    let mu_num: f64 = data
        .iter()
        .zip(responsibilities)
        .map(|(&value, &r)| r * value)
        .sum();
    let mu_signal = if sum_r > 1e-6 {
        mu_num / sum_r
    } else {
        data.iter().copied().fold(0.0, f64::max)
    };

    let var_num: f64 = data
        .iter()
        .zip(responsibilities)
        .map(|(&value, &r)| {
            let centered = value - mu_signal;
            r * centered * centered
        })
        .sum();
    let sigma_signal = (var_num / sum_r.max(1e-6)).sqrt().max(0.25);

    FitParams {
        pi,
        lambda_bg,
        mu_signal,
        sigma_signal,
    }
}

struct ParsedInput {
    cell_order: Vec<String>,
    guide_order: Vec<String>,
    counts_by_guide: HashMap<String, Vec<u32>>,
}

impl ParsedInput {
    fn from_assignment_input(input: &AssignmentInput) -> Result<Self> {
        let counts = &input.counts;
        let barcodes = counts
            .column_by_name("cell_barcode")
            .context("missing cell_barcode")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("cell_barcode not Utf8")?;
        let guide_ids = counts
            .column_by_name("guide_id")
            .context("missing guide_id")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("guide_id not Utf8")?;
        let umi_counts = counts
            .column_by_name("umi_count")
            .context("missing umi_count")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("umi_count not UInt32")?;

        let mut cell_order = covariate_cell_order(input)?;
        let mut cell_index: HashMap<String, usize> = cell_order
            .iter()
            .enumerate()
            .map(|(idx, cell)| (cell.clone(), idx))
            .collect();

        let mut guide_order: Vec<String> = Vec::new();
        let mut counts_by_guide: HashMap<String, Vec<u32>> = HashMap::new();

        for row_idx in 0..counts.num_rows() {
            let barcode = barcodes.value(row_idx);
            let cell_idx = match cell_index.get(barcode) {
                Some(&idx) => idx,
                None => {
                    let idx = cell_order.len();
                    cell_order.push(barcode.to_string());
                    cell_index.insert(barcode.to_string(), idx);
                    for guide_counts in counts_by_guide.values_mut() {
                        guide_counts.push(0);
                    }
                    idx
                }
            };

            let guide = guide_ids.value(row_idx).to_string();
            let guide_counts = counts_by_guide.entry(guide.clone()).or_insert_with(|| {
                guide_order.push(guide.clone());
                vec![0; cell_order.len()]
            });
            if guide_counts.len() < cell_order.len() {
                guide_counts.resize(cell_order.len(), 0);
            }
            guide_counts[cell_idx] = guide_counts[cell_idx].saturating_add(umi_counts.value(row_idx));
        }

        Ok(Self {
            cell_order,
            guide_order,
            counts_by_guide,
        })
    }
}

fn covariate_cell_order(input: &AssignmentInput) -> Result<Vec<String>> {
    if input.covariates.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let barcodes = input
        .covariates
        .column_by_name("cell_barcode")
        .context("missing covariate cell_barcode")?
        .as_any()
        .downcast_ref::<StringArray>()
        .context("covariate cell_barcode not Utf8")?;

    Ok((0..input.covariates.num_rows())
        .map(|idx| barcodes.value(idx).to_string())
        .collect())
}

fn clamp_probability(value: f64) -> f64 {
    value.clamp(1e-6, 1.0 - 1e-6)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{BooleanArray, Float32Array},
        datatypes::{DataType, Field, Schema},
    };

    fn make_counts(rows: Vec<(&str, &str, u32)>) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_barcode", DataType::Utf8, false),
            Field::new("guide_id", DataType::Utf8, false),
            Field::new("umi_count", DataType::UInt32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    rows.iter().map(|(cell, _, _)| *cell).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    rows.iter().map(|(_, guide, _)| *guide).collect::<Vec<_>>(),
                )),
                Arc::new(UInt32Array::from(
                    rows.iter().map(|(_, _, count)| *count).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    fn covariates(cells: Vec<&str>) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_barcode", DataType::Utf8, false),
            Field::new("batch", DataType::Utf8, false),
            Field::new("total_counts", DataType::Float32, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(cells.clone())),
                Arc::new(StringArray::from(vec!["default"; cells.len()])),
                Arc::new(Float32Array::from(vec![0.0; cells.len()])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn assigns_high_signal_cells() {
        let counts = make_counts(vec![
            ("C1", "gA", 0),
            ("C2", "gA", 1),
            ("C3", "gA", 0),
            ("C4", "gA", 25),
            ("C5", "gA", 30),
            ("C6", "gA", 22),
        ]);
        let input = AssignmentInput {
            counts,
            covariates: covariates(vec!["C1", "C2", "C3", "C4", "C5", "C6"]),
        };
        let model = PoissonGaussModel {
            min_confidence: 0.5,
            ..Default::default()
        };

        let result = model.assign(&input).unwrap();
        let is_unassigned = result
            .column_by_name("is_unassigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        assert!(is_unassigned.value(0));
        assert!(is_unassigned.value(1));
        assert!(!is_unassigned.value(3));
        assert!(!is_unassigned.value(4));
        assert!(!is_unassigned.value(5));
    }

    #[test]
    fn skips_guides_with_too_little_signal() {
        let counts = make_counts(vec![("C1", "gA", 0), ("C2", "gA", 1), ("C3", "gA", 1)]);
        let input = AssignmentInput {
            counts,
            covariates: covariates(vec!["C1", "C2", "C3"]),
        };

        let result = PoissonGaussModel::default().assign(&input).unwrap();
        let is_unassigned = result
            .column_by_name("is_unassigned")
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();

        assert!(is_unassigned.value(0));
        assert!(is_unassigned.value(1));
        assert!(is_unassigned.value(2));
    }
}

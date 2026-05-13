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
use serde_json::{json, Value};
use std::sync::Arc;

/// Assign each cell to the guide with the largest fraction of total guide UMIs,
/// if that fraction exceeds min_fraction.
///
/// Per cell:
/// - top_fraction = top_guide_umi / sum(all guide UMIs for cell)
/// - top_fraction > min_fraction → assign top guide; confidence = top_fraction
/// - Otherwise → is_unassigned
pub struct RatioModel {
    pub min_fraction: f32,
}

impl Default for RatioModel {
    fn default() -> Self {
        Self { min_fraction: 0.3 }
    }
}

impl AssignmentModel for RatioModel {
    fn name(&self) -> &'static str {
        "ratio"
    }

    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult> {
        let n_cells = input.counts.n_cells;
        let n_guides = input.counts.n_guides;
        let csr = input.counts.csr();
        let guide_ids = &input.guide_metadata.guide_ids;
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

            let total: u32 = row.values().iter().sum();

            if total == 0 {
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                out_n_detected.append_value(0);
                continue;
            }

            let (top_g, top_count) = row.col_indices().iter().zip(row.values())
                .max_by_key(|(_, &v)| v)
                .map(|(&g, &v)| (g, v))
                .unwrap();

            let fraction = top_count as f32 / total as f32;

            if fraction > self.min_fraction {
                out_n_detected.append_value(1);
                out_guide_ids.append_value(guide_ids.value(top_g));
                out_target_genes.append_null();
                out_umi_counts.append_value(top_count);
                out_confidence.append_value(fraction);
                out_is_unassigned.append_value(false);
                out_is_multi.append_value(false);
                assigned_triples.push((cell, top_g));
            } else {
                out_n_detected.append_value(0);
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
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
        let assigned_x = build_assigned_csr(assigned_triples, n_cells, n_guides);

        Ok(AssignmentResult { batch, assigned_x })
    }

    fn params_json(&self) -> Value {
        json!({ "min_fraction": self.min_fraction })
    }
}

fn build_assigned_csr(triples: Vec<(usize, usize)>, n_cells: usize, n_guides: usize) -> CsrMatrix<u8> {
    let nnz = triples.len();
    let mut row_offsets = vec![0usize; n_cells + 1];
    let mut col_indices = Vec::with_capacity(nnz);
    let mut last = 0usize;
    for (idx, &(r, c)) in triples.iter().enumerate() {
        while last < r { row_offsets[last + 1] = idx; last += 1; }
        col_indices.push(c);
    }
    for i in (last + 1)..=n_cells { row_offsets[i] = nnz; }
    CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, vec![1u8; nnz])
        .expect("assigned CSR from sorted triples cannot fail")
}

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
                total_counts: Float32Array::from(vec![0.0f32; n_cells]),
            },
            guide_metadata: GuideMetadata { guide_ids: gd.finish() },
        }
    }

    fn is_unassigned(r: &AssignmentResult) -> &BooleanArray {
        r.batch.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap()
    }

    #[test]
    fn dominant_guide_assigned() {
        // g0=90, g1=10 → fraction 0.9 > 0.3 → assign g0
        let input = make_input(1, 2, vec![(0, 0, 90), (0, 1, 10)]);
        let result = RatioModel::default().assign(&input).unwrap();
        assert!(!is_unassigned(&result).value(0));
    }

    #[test]
    fn mixed_guide_unassigned() {
        // 50/50 split → top fraction = 0.5, which is not > 0.5 threshold
        let model = RatioModel { min_fraction: 0.5 };
        let input = make_input(1, 2, vec![(0, 0, 5), (0, 1, 5)]);
        let result = model.assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn fraction_exactly_at_threshold_is_unassigned() {
        // Condition is strictly >. g0=4, g1=4, g2=2 → top fraction = 4/10 = 0.4, not > 0.4.
        let model = RatioModel { min_fraction: 0.4 };
        let input = make_input(1, 3, vec![(0, 0, 4), (0, 1, 4), (0, 2, 2)]);
        let result = model.assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0), "fraction == threshold should be unassigned");
    }

    #[test]
    fn n_detected_zero_for_below_threshold() {
        // Unassigned cell should have n_detected=0, not 1
        let model = RatioModel { min_fraction: 0.9 };
        let input = make_input(1, 2, vec![(0, 0, 5), (0, 1, 5)]);
        let result = model.assign(&input).unwrap();
        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();
        assert!(is_unassigned(&result).value(0));
        assert_eq!(n_det.value(0), 0, "below-threshold cell must have n_detected=0");
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        let input = make_input(1, 2, vec![(0, 0, 5), (0, 1, 5)]);
        let model = RatioModel { min_fraction: 0.9 };
        let result = model.assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
        assert!(result.batch.column_by_name("guide_id").unwrap().is_null(0));
        assert!(result.batch.column_by_name("umi_count").unwrap().is_null(0));
        assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(0));
    }

    #[test]
    fn zero_counts_unassigned() {
        let input = make_input(1, 2, vec![]);
        let result = RatioModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn confidence_equals_fraction() {
        // g0=80, g1=20 → fraction = 0.8
        let input = make_input(1, 2, vec![(0, 0, 80), (0, 1, 20)]);
        let result = RatioModel::default().assign(&input).unwrap();
        use arrow::array::Float32Array;
        let conf = result.batch.column_by_name("assignment_confidence").unwrap()
            .as_any().downcast_ref::<Float32Array>().unwrap();
        assert!((conf.value(0) - 0.8f32).abs() < 1e-5);
    }

    #[test]
    fn single_guide_always_assigned() {
        // Only one guide with any count → fraction = 1.0 > any threshold
        let input = make_input(1, 1, vec![(0, 0, 7)]);
        let result = RatioModel::default().assign(&input).unwrap();
        assert!(!is_unassigned(&result).value(0));
    }

    #[test]
    fn assigned_x_correct() {
        let input = make_input(2, 2, vec![(0, 0, 80), (0, 1, 20), (1, 1, 50)]);
        let result = RatioModel::default().assign(&input).unwrap();
        assert_eq!(result.assigned_x.get_row(0).unwrap().nnz(), 1);
        assert_eq!(result.assigned_x.get_row(0).unwrap().col_indices(), &[0]);
        assert_eq!(result.assigned_x.get_row(1).unwrap().nnz(), 1);
        assert_eq!(result.assigned_x.get_row(1).unwrap().col_indices(), &[1]);
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = RatioModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }
}

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

/// Assign each cell-guide pair as positive if UMI count >= umi_threshold.
///
/// - 0 guides above threshold → is_unassigned
/// - 1 guide above threshold → assign, confidence = 1.0
/// - 2+ guides above threshold → is_multi_infected; assign to highest UMI
pub struct UmiModel {
    pub umi_threshold: u32,
}

impl Default for UmiModel {
    fn default() -> Self {
        Self { umi_threshold: 5 }
    }
}

impl AssignmentModel for UmiModel {
    fn name(&self) -> &'static str {
        "umi"
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

            // Collect (guide_idx, count) for entries above threshold.
            let above: Vec<(usize, u32)> = row.col_indices().iter().zip(row.values())
                .filter(|(_, &v)| v >= self.umi_threshold)
                .map(|(&g, &v)| (g, v))
                .collect();

            out_n_detected.append_value(above.len().min(255) as u8);

            match above.len() {
                0 => {
                    out_guide_ids.append_null();
                    out_target_genes.append_null();
                    out_umi_counts.append_null();
                    out_confidence.append_null();
                    out_is_unassigned.append_value(true);
                    out_is_multi.append_value(false);
                }
                1 => {
                    let (g, count) = above[0];
                    out_guide_ids.append_value(guide_ids.value(g));
                    out_target_genes.append_null();
                    out_umi_counts.append_value(count);
                    out_confidence.append_value(1.0);
                    out_is_unassigned.append_value(false);
                    out_is_multi.append_value(false);
                    assigned_triples.push((cell, g));
                }
                _ => {
                    // Multi-infected: assign to highest UMI guide.
                    let (g, count) = above.iter().copied().max_by_key(|&(_, v)| v).unwrap();
                    out_guide_ids.append_value(guide_ids.value(g));
                    out_target_genes.append_null();
                    out_umi_counts.append_value(count);
                    out_confidence.append_value(1.0);
                    out_is_unassigned.append_value(false);
                    out_is_multi.append_value(true);
                    for &(ag, _) in &above {
                        assigned_triples.push((cell, ag));
                    }
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
        json!({ "umi_threshold": self.umi_threshold })
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
        .expect("assigned CSR from sorted unique triples cannot fail")
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
    fn single_guide_above_threshold() {
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 1)]);
        let result = UmiModel::default().assign(&input).unwrap();
        let isu = is_unassigned(&result);
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!isu.value(0));
        assert!(!is_multi.value(0));
    }

    #[test]
    fn no_guides_above_threshold_is_unassigned() {
        let input = make_input(1, 2, vec![(0, 0, 2), (0, 1, 1)]);
        let result = UmiModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn two_guides_above_threshold_is_multi() {
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 8)]);
        let result = UmiModel::default().assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(0));
    }

    #[test]
    fn multi_assigns_to_highest_umi() {
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 8)]);
        let result = UmiModel::default().assign(&input).unwrap();
        use arrow::array::{DictionaryArray, StringArray as SA};
        use arrow::datatypes::Int16Type;
        let dict = result.batch.column_by_name("guide_id").unwrap()
            .as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
        let vals = dict.values().as_any().downcast_ref::<SA>().unwrap();
        let key = dict.keys().value(0) as usize;
        assert_eq!(vals.value(key), "g0"); // g0 has count 10 > g1 count 8
    }

    #[test]
    fn assigned_x_covers_all_above_threshold_for_multi() {
        // both guides above threshold → both in assigned_x
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 8)]);
        let result = UmiModel::default().assign(&input).unwrap();
        assert_eq!(result.assigned_x.get_row(0).unwrap().nnz(), 2);
    }

    #[test]
    fn n_detected_counts_above_threshold() {
        let input = make_input(1, 3, vec![(0, 0, 10), (0, 1, 8), (0, 2, 2)]);
        let result = UmiModel::default().assign(&input).unwrap();
        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(n_det.value(0), 2); // g0=10 and g1=8 above threshold=5; g2=2 below
    }

    #[test]
    fn empty_matrix_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = UmiModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn custom_threshold_zero_assigns_any_nonzero() {
        let model = UmiModel { umi_threshold: 1 };
        let input = make_input(1, 2, vec![(0, 0, 1)]);
        let result = model.assign(&input).unwrap();
        assert!(!is_unassigned(&result).value(0));
    }

    #[test]
    fn count_exactly_at_threshold_is_assigned() {
        // Filter is >=, so count == umi_threshold must pass.
        let model = UmiModel { umi_threshold: 5 };
        let input = make_input(1, 1, vec![(0, 0, 5)]);
        let result = model.assign(&input).unwrap();
        assert!(!is_unassigned(&result).value(0));
    }

    #[test]
    fn count_one_below_threshold_is_unassigned() {
        let model = UmiModel { umi_threshold: 5 };
        let input = make_input(1, 1, vec![(0, 0, 4)]);
        let result = model.assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn n_detected_zero_for_unassigned_cell() {
        let input = make_input(1, 2, vec![(0, 0, 1), (0, 1, 1)]); // both below threshold=5
        let result = UmiModel::default().assign(&input).unwrap();
        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();
        assert!(is_unassigned(&result).value(0));
        assert_eq!(n_det.value(0), 0);
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        let input = make_input(1, 2, vec![(0, 0, 1)]); // below threshold=5
        let result = UmiModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
        assert!(result.batch.column_by_name("guide_id").unwrap().is_null(0));
        assert!(result.batch.column_by_name("umi_count").unwrap().is_null(0));
        assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(0));
    }
}

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

/// Assign each cell to the guide with the highest UMI count.
///
/// - All-zero counts → is_unassigned
/// - Unique highest guide above umi_threshold → assign, confidence = 1.0
/// - Tie for highest → is_unassigned
pub struct MaxModel {
    pub umi_threshold: u32,
}

impl Default for MaxModel {
    fn default() -> Self {
        Self { umi_threshold: 0 }
    }
}

impl AssignmentModel for MaxModel {
    fn name(&self) -> &'static str {
        "max"
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

            let max_count = row.values().iter().copied().max().unwrap_or(0);

            if max_count == 0 || max_count < self.umi_threshold {
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                out_n_detected.append_value(0);
                continue;
            }

            let top: Vec<usize> = row.col_indices().iter().zip(row.values())
                .filter(|(_, &v)| v == max_count)
                .map(|(&g, _)| g)
                .collect();

            out_n_detected.append_value(top.len().min(255) as u8);

            if top.len() > 1 {
                // Tie → unassigned.
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
            } else {
                let g = top[0];
                out_guide_ids.append_value(guide_ids.value(g));
                out_target_genes.append_null();
                out_umi_counts.append_value(max_count);
                out_confidence.append_value(1.0);
                out_is_unassigned.append_value(false);
                out_is_multi.append_value(false);
                assigned_triples.push((cell, g));
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
    use arrow::array::{BooleanArray, Float32Array, StringBuilder, UInt32Array};

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
    fn assigns_max_guide() {
        // Cell 0: g0=15, g1=5 → assign g0
        let input = make_input(1, 2, vec![(0, 0, 15), (0, 1, 5)]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert!(!is_unassigned(&result).value(0));

        let umi = result.batch.column_by_name("umi_count").unwrap()
            .as_any().downcast_ref::<UInt32Array>().unwrap();
        assert_eq!(umi.value(0), 15);
    }

    #[test]
    fn tie_is_unassigned() {
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 10)]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn all_zero_is_unassigned() {
        // No nonzero entries for cell 0
        let input = make_input(1, 2, vec![]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn umi_threshold_filters_low_counts() {
        let input = make_input(1, 2, vec![(0, 0, 3), (0, 1, 1)]);
        let model = MaxModel { umi_threshold: 5 };
        let result = model.assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
    }

    #[test]
    fn multi_cell_correct_assignments() {
        // C0: g0=10, g1=2  → g0
        // C1: g0=5,  g1=5  → tie → unassigned
        // C2: g1=8         → g1
        let input = make_input(3, 2, vec![
            (0, 0, 10), (0, 1, 2),
            (1, 0, 5), (1, 1, 5),
            (2, 1, 8),
        ]);
        let result = MaxModel::default().assign(&input).unwrap();
        let isu = is_unassigned(&result);
        assert!(!isu.value(0));
        assert!(isu.value(1));
        assert!(!isu.value(2));
    }

    #[test]
    fn guide_name_in_output_correct() {
        let input = make_input(1, 2, vec![(0, 1, 20)]);
        let result = MaxModel::default().assign(&input).unwrap();
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int16Type;
        let dict = result.batch.column_by_name("guide_id").unwrap()
            .as_any().downcast_ref::<DictionaryArray<Int16Type>>().unwrap();
        let vals = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        let key = dict.keys().value(0) as usize;
        assert_eq!(vals.value(key), "g1");
    }

    #[test]
    fn assigned_x_set_for_assigned_cells() {
        let input = make_input(2, 2, vec![(0, 0, 10), (1, 1, 5)]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert_eq!(result.assigned_x.get_row(0).unwrap().nnz(), 1);
        assert_eq!(result.assigned_x.get_row(1).unwrap().nnz(), 1);
    }

    #[test]
    fn assigned_x_empty_for_unassigned_cells() {
        let input = make_input(1, 2, vec![(0, 0, 5), (0, 1, 5)]); // tie
        let result = MaxModel::default().assign(&input).unwrap();
        assert_eq!(result.assigned_x.get_row(0).unwrap().nnz(), 0);
    }

    #[test]
    fn empty_input_produces_empty_result() {
        let input = make_input(0, 0, vec![]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert_eq!(result.batch.num_rows(), 0);
    }

    #[test]
    fn confidence_is_one_for_assigned() {
        let input = make_input(1, 2, vec![(0, 0, 15), (0, 1, 3)]);
        let result = MaxModel::default().assign(&input).unwrap();
        use arrow::array::Float32Array;
        let conf = result.batch.column_by_name("assignment_confidence").unwrap()
            .as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(conf.value(0), 1.0);
    }

    #[test]
    fn tie_n_detected_is_two_not_one() {
        // Two guides tied for max — n_detected reflects count of tied guides, cell is unassigned.
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 10)]);
        let result = MaxModel::default().assign(&input).unwrap();
        use arrow::array::UInt8Array;
        let n_det = result.batch.column_by_name("n_guides_detected").unwrap()
            .as_any().downcast_ref::<UInt8Array>().unwrap();
        assert!(is_unassigned(&result).value(0));
        assert_eq!(n_det.value(0), 2);
    }

    #[test]
    fn is_multi_never_set_by_max_model() {
        // MaxModel resolves ties by returning unassigned, never sets is_multi_infected.
        let input = make_input(1, 2, vec![(0, 0, 10), (0, 1, 10)]);
        let result = MaxModel::default().assign(&input).unwrap();
        let is_multi = result.batch.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!is_multi.value(0));
    }

    #[test]
    fn unassigned_columns_are_null() {
        use arrow::array::Array;
        // All-zero → unassigned
        let input = make_input(1, 2, vec![]);
        let result = MaxModel::default().assign(&input).unwrap();
        assert!(is_unassigned(&result).value(0));
        assert!(result.batch.column_by_name("guide_id").unwrap().is_null(0));
        assert!(result.batch.column_by_name("umi_count").unwrap().is_null(0));
        assert!(result.batch.column_by_name("assignment_confidence").unwrap().is_null(0));
    }
}

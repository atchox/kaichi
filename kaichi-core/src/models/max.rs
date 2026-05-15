use super::AssignmentModel;
use super::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use serde_json::{json, Value};

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

        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let row = csr.get_row(cell).unwrap();
            let max_count = row.values().iter().copied().max().unwrap_or(0);

            if max_count == 0 || max_count < self.umi_threshold {
                out.append_unassigned(0);
                continue;
            }

            // Collect all guide indices tied for max — at most n_guides, typically 1.
            let mut top_idx: usize = 0;
            let mut n_tied: usize = 0;
            for (&g, &v) in row.col_indices().iter().zip(row.values()) {
                if v == max_count {
                    top_idx = g;
                    n_tied += 1;
                }
            }

            let n_det = n_detected_u8(n_tied);
            if n_tied > 1 {
                out.append_unassigned(n_det);
            } else {
                out.append_assigned(guide_ids.value(top_idx), max_count, 1.0, false, n_det);
                out.push_assigned_triple(cell, top_idx);
            }
        }

        // Triples emitted in cell-major order with one entry per cell — already sorted.
        out.finish(cell_barcodes, true)
    }

    fn params_json(&self) -> Value {
        json!({ "umi_threshold": self.umi_threshold })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_support::input_zero_totals as make_input;
    use arrow::array::{BooleanArray, StringArray, UInt32Array};

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

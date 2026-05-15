use super::AssignmentModel;
use super::output::AssignmentOutputBuilder;
use crate::data::{AssignmentResult, LoadedInput};

use anyhow::Result;
use serde_json::{json, Value};

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

        let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, self.name());

        for cell in 0..n_cells {
            let row = csr.get_row(cell).unwrap();

            // Single pass: sum and argmax in one go.
            let mut total: u32 = 0;
            let mut best: Option<(usize, u32)> = None;
            for (&g, &v) in row.col_indices().iter().zip(row.values()) {
                total += v;
                match best {
                    None => best = Some((g, v)),
                    Some((_, bv)) if v > bv => best = Some((g, v)),
                    _ => {}
                }
            }

            if total == 0 {
                out.append_unassigned(0);
                continue;
            }

            let (top_g, top_count) = best.unwrap();
            let fraction = top_count as f32 / total as f32;

            if fraction > self.min_fraction {
                out.append_assigned(guide_ids.value(top_g), top_count, fraction, false, 1);
                out.push_assigned_triple(cell, top_g);
            } else {
                out.append_unassigned(0);
            }
        }

        // Triples emitted in cell-major order with at most one entry per cell — sorted.
        out.finish(cell_barcodes, true)
    }

    fn params_json(&self) -> Value {
        json!({ "min_fraction": self.min_fraction })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::test_support::input_zero_totals as make_input;
    use arrow::array::BooleanArray;

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

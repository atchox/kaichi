use super::{AssignmentInput, AssignmentModel};
use crate::schema::assignment_schema_ref;

use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, StringArray, StringDictionaryBuilder,
        UInt32Array, UInt32Builder, UInt8Builder, Float32Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Assign each cell-guide pair as positive if UMI count >= umi_threshold.
///
/// - 0 guides above threshold → is_unassigned
/// - 1 guide above threshold → assign
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

    fn assign(&self, input: &AssignmentInput) -> Result<RecordBatch> {
        let counts = &input.counts;

        let barcodes = counts
            .column_by_name("cell_barcode")
            .context("missing cell_barcode column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("cell_barcode is not Utf8")?;

        let guide_ids = counts
            .column_by_name("guide_id")
            .context("missing guide_id column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("guide_id is not Utf8")?;

        let umi_counts = counts
            .column_by_name("umi_count")
            .context("missing umi_count column")?
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("umi_count is not UInt32")?;

        // Group rows by cell barcode.
        let mut cell_guides: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();
        // Maintain insertion order for deterministic output.
        let mut cell_order: Vec<&str> = Vec::new();

        for i in 0..counts.num_rows() {
            let barcode = barcodes.value(i);
            let guide = guide_ids.value(i);
            let count = umi_counts.value(i);
            let entry = cell_guides.entry(barcode).or_insert_with(|| {
                cell_order.push(barcode);
                Vec::new()
            });
            entry.push((guide, count));
        }

        let n = cell_order.len();
        let mut out_barcodes: Vec<&str> = Vec::with_capacity(n);
        let mut out_guide_ids: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut out_target_genes: StringDictionaryBuilder<Int16Type> =
            StringDictionaryBuilder::new();
        let mut out_umi_counts = UInt32Builder::with_capacity(n);
        let mut out_confidence = Float32Builder::with_capacity(n);
        let mut out_is_unassigned = BooleanBuilder::with_capacity(n);
        let mut out_is_multi = BooleanBuilder::with_capacity(n);
        let mut out_n_detected = UInt8Builder::with_capacity(n);

        for barcode in &cell_order {
            let guides = &cell_guides[barcode];

            let above: Vec<(&str, u32)> = guides
                .iter()
                .filter(|(_, c)| *c >= self.umi_threshold)
                .cloned()
                .collect();

            let n_detected = above.len().min(255) as u8;
            out_barcodes.push(barcode);
            out_n_detected.append_value(n_detected);

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
                    let (guide, count) = above[0];
                    out_guide_ids.append_value(guide);
                    out_target_genes.append_value(guide); // placeholder; real target_gene comes from guide library join
                    out_umi_counts.append_value(count);
                    out_confidence.append_value(1.0);
                    out_is_unassigned.append_value(false);
                    out_is_multi.append_value(false);
                }
                _ => {
                    // Multi-infected: assign to highest UMI guide.
                    let (guide, count) = above
                        .iter()
                        .max_by_key(|(_, c)| c)
                        .copied()
                        .unwrap();
                    out_guide_ids.append_value(guide);
                    out_target_genes.append_value(guide);
                    out_umi_counts.append_value(count);
                    out_confidence.append_value(1.0);
                    out_is_unassigned.append_value(false);
                    out_is_multi.append_value(true);
                }
            }
        }

        let model_name_arr: DictionaryArray<arrow::datatypes::Int8Type> = {
            let values = StringArray::from(vec![self.name()]);
            let keys = arrow::array::Int8Array::from(vec![0i8; n]);
            DictionaryArray::try_new(keys, std::sync::Arc::new(values))
                .context("failed to build assignment_model dictionary")?
        };

        RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                std::sync::Arc::new(StringArray::from(out_barcodes)),
                std::sync::Arc::new(out_guide_ids.finish()),
                std::sync::Arc::new(out_target_genes.finish()),
                std::sync::Arc::new(out_umi_counts.finish()),
                std::sync::Arc::new(out_confidence.finish()),
                std::sync::Arc::new(model_name_arr),
                std::sync::Arc::new(out_is_unassigned.finish()),
                std::sync::Arc::new(out_is_multi.finish()),
                std::sync::Arc::new(out_n_detected.finish()),
            ],
        )
        .context("failed to build output RecordBatch")
    }

    fn params_json(&self) -> Value {
        json!({ "umi_threshold": self.umi_threshold })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn make_counts(rows: Vec<(&str, &str, u32)>) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_barcode", DataType::Utf8, false),
            Field::new("guide_id", DataType::Utf8, false),
            Field::new("umi_count", DataType::UInt32, false),
        ]);
        let barcodes = StringArray::from(rows.iter().map(|(b, _, _)| *b).collect::<Vec<_>>());
        let guides = StringArray::from(rows.iter().map(|(_, g, _)| *g).collect::<Vec<_>>());
        let counts = UInt32Array::from(rows.iter().map(|(_, _, c)| *c).collect::<Vec<_>>());
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(barcodes), Arc::new(guides), Arc::new(counts)],
        )
        .unwrap()
    }

    fn empty_covariates() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_barcode", DataType::Utf8, false),
            Field::new("batch", DataType::Utf8, false),
            Field::new("total_counts", DataType::Float32, false),
        ]);
        RecordBatch::new_empty(Arc::new(schema))
    }

    #[test]
    fn single_guide_above_threshold() {
        let model = UmiModel::default();
        let counts = make_counts(vec![("CELL1", "guide_A", 10), ("CELL1", "guide_B", 1)]);
        let input = AssignmentInput { counts, covariates: empty_covariates() };
        let result = model.assign(&input).unwrap();

        assert_eq!(result.num_rows(), 1);
        let is_unassigned = result
            .column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let is_multi = result
            .column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();

        assert!(!is_unassigned.value(0));
        assert!(!is_multi.value(0));
    }

    #[test]
    fn no_guides_above_threshold_is_unassigned() {
        let model = UmiModel::default();
        let counts = make_counts(vec![("CELL1", "guide_A", 2), ("CELL1", "guide_B", 1)]);
        let input = AssignmentInput { counts, covariates: empty_covariates() };
        let result = model.assign(&input).unwrap();

        let is_unassigned = result
            .column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_unassigned.value(0));
    }

    #[test]
    fn two_guides_above_threshold_is_multi() {
        let model = UmiModel::default();
        let counts = make_counts(vec![("CELL1", "guide_A", 10), ("CELL1", "guide_B", 8)]);
        let input = AssignmentInput { counts, covariates: empty_covariates() };
        let result = model.assign(&input).unwrap();

        let is_multi = result
            .column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_multi.value(0));
    }
}

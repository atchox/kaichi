use super::{AssignmentInput, AssignmentModel};
use crate::schema::assignment_schema_ref;

use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, Float32Builder, StringArray,
        StringDictionaryBuilder, UInt32Array, UInt32Builder, UInt8Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};

/// Assign each cell to the guide with the highest UMI count.
///
/// - All-zero counts → is_unassigned
/// - Single highest guide → assign
/// - Tie for highest → is_unassigned
pub struct MaxModel {
    /// Optional post-hoc filter: assigned guide must have >= umi_threshold UMIs.
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

    fn assign(&self, input: &AssignmentInput) -> Result<RecordBatch> {
        let counts = &input.counts;

        let barcodes = counts
            .column_by_name("cell_barcode").context("missing cell_barcode")?
            .as_any().downcast_ref::<StringArray>().context("cell_barcode not Utf8")?;
        let guide_ids = counts
            .column_by_name("guide_id").context("missing guide_id")?
            .as_any().downcast_ref::<StringArray>().context("guide_id not Utf8")?;
        let umi_counts = counts
            .column_by_name("umi_count").context("missing umi_count")?
            .as_any().downcast_ref::<UInt32Array>().context("umi_count not UInt32")?;

        let mut cell_guides: HashMap<&str, Vec<(&str, u32)>> = HashMap::new();
        let mut cell_order: Vec<&str> = Vec::new();

        for i in 0..counts.num_rows() {
            let barcode = barcodes.value(i);
            let entry = cell_guides.entry(barcode).or_insert_with(|| {
                cell_order.push(barcode);
                Vec::new()
            });
            entry.push((guide_ids.value(i), umi_counts.value(i)));
        }

        let n = cell_order.len();
        let mut out_barcodes: Vec<&str> = Vec::with_capacity(n);
        let mut out_guide_ids: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        let mut out_target_genes: StringDictionaryBuilder<Int16Type> = StringDictionaryBuilder::new();
        let mut out_umi_counts = UInt32Builder::with_capacity(n);
        let mut out_confidence = Float32Builder::with_capacity(n);
        let mut out_is_unassigned = BooleanBuilder::with_capacity(n);
        let mut out_is_multi = BooleanBuilder::with_capacity(n);
        let mut out_n_detected = UInt8Builder::with_capacity(n);

        for barcode in &cell_order {
            let guides = &cell_guides[barcode];
            let max_count = guides.iter().map(|(_, c)| c).max().copied().unwrap_or(0);

            out_barcodes.push(barcode);

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

            let top: Vec<&str> = guides
                .iter()
                .filter(|(_, c)| *c == max_count)
                .map(|(g, _)| *g)
                .collect();

            if top.len() > 1 {
                // Tie → unassigned.
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                out_n_detected.append_value(top.len().min(255) as u8);
            } else {
                out_guide_ids.append_value(top[0]);
                out_target_genes.append_value(top[0]);
                out_umi_counts.append_value(max_count);
                out_confidence.append_value(1.0);
                out_is_unassigned.append_value(false);
                out_is_multi.append_value(false);
                out_n_detected.append_value(1);
            }
        }

        let model_name_arr: DictionaryArray<arrow::datatypes::Int8Type> = {
            let values = StringArray::from(vec![self.name()]);
            let keys = arrow::array::Int8Array::from(vec![0i8; n]);
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
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(rows.iter().map(|(b, _, _)| *b).collect::<Vec<_>>())),
                Arc::new(StringArray::from(rows.iter().map(|(_, g, _)| *g).collect::<Vec<_>>())),
                Arc::new(UInt32Array::from(rows.iter().map(|(_, _, c)| *c).collect::<Vec<_>>())),
            ],
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
    fn assigns_max_guide() {
        let model = MaxModel::default();
        let counts = make_counts(vec![("C1", "gA", 15), ("C1", "gB", 5)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!is_unassigned.value(0));
    }

    #[test]
    fn tie_is_unassigned() {
        let model = MaxModel::default();
        let counts = make_counts(vec![("C1", "gA", 10), ("C1", "gB", 10)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_unassigned.value(0));
    }

    #[test]
    fn all_zero_is_unassigned() {
        let model = MaxModel::default();
        let counts = make_counts(vec![("C1", "gA", 0), ("C1", "gB", 0)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_unassigned.value(0));
    }
}

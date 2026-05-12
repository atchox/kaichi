use super::{AssignmentInput, AssignmentModel};
use crate::schema::assignment_schema_ref;

use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, Float32Builder, StringArray,
        StringDictionaryBuilder, UInt32Builder, UInt8Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};

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
            .as_any().downcast_ref::<arrow::array::UInt32Array>().context("umi_count not UInt32")?;

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
            out_barcodes.push(barcode);

            let total_umi: u32 = guides.iter().map(|(_, c)| *c).sum();

            if total_umi == 0 {
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                out_n_detected.append_value(0);
                continue;
            }

            let (top_guide, top_count) = guides.iter()
                .max_by_key(|(_, c)| *c)
                .copied()
                .unwrap();

            let top_fraction = top_count as f32 / total_umi as f32;

            if top_fraction > self.min_fraction {
                out_guide_ids.append_value(top_guide);
                out_target_genes.append_value(top_guide);
                out_umi_counts.append_value(top_count);
                out_confidence.append_value(top_fraction);
                out_is_unassigned.append_value(false);
                out_is_multi.append_value(false);
                let n_nonzero: u8 = guides.iter().filter(|(_, c)| *c > 0).count().min(255) as u8;
                out_n_detected.append_value(n_nonzero);
            } else {
                out_guide_ids.append_null();
                out_target_genes.append_null();
                out_umi_counts.append_null();
                out_confidence.append_null();
                out_is_unassigned.append_value(true);
                out_is_multi.append_value(false);
                let n_nonzero: u8 = guides.iter().filter(|(_, c)| *c > 0).count().min(255) as u8;
                out_n_detected.append_value(n_nonzero);
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
        json!({ "min_fraction": self.min_fraction })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Float32Array, StringArray, UInt32Array};
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
    fn dominant_guide_assigned() {
        // gA = 30, gB = 5 → total = 35, fraction = 30/35 ≈ 0.857 > 0.3 → assigned
        let model = RatioModel::default();
        let counts = make_counts(vec![("C1", "gA", 30), ("C1", "gB", 5)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let is_multi = result.column_by_name("is_multi_infected").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        let confidence = result.column_by_name("assignment_confidence").unwrap()
            .as_any().downcast_ref::<Float32Array>().unwrap();
        assert!(!is_unassigned.value(0));
        assert!(!is_multi.value(0));
        assert!((confidence.value(0) - 30.0 / 35.0).abs() < 1e-5);
    }

    #[test]
    fn low_fraction_unassigned() {
        // 4 guides × 2 UMI each → total = 8, top_fraction = 2/8 = 0.25 < 0.3 → unassigned
        let model = RatioModel::default();
        let counts = make_counts(vec![
            ("C1", "gA", 2), ("C1", "gB", 2), ("C1", "gC", 2), ("C1", "gD", 2),
        ]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_unassigned.value(0));
    }

    #[test]
    fn single_guide_above_threshold_assigned() {
        // gA = 10, total = 10, fraction = 1.0 > 0.3 → assigned
        let model = RatioModel::default();
        let counts = make_counts(vec![("C1", "gA", 10)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!is_unassigned.value(0));
    }

    #[test]
    fn all_zero_unassigned() {
        let model = RatioModel::default();
        let counts = make_counts(vec![("C1", "gA", 0), ("C1", "gB", 0)]);
        let result = model.assign(&AssignmentInput { counts, covariates: empty_covariates() }).unwrap();
        let is_unassigned = result.column_by_name("is_unassigned").unwrap()
            .as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(is_unassigned.value(0));
    }
}

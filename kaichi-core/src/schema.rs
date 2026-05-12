use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Arrow schema for the per-cell guide call RecordBatch returned by all models.
pub fn assignment_schema() -> Schema {
    Schema::new(vec![
        Field::new("cell_barcode", DataType::Utf8, false),
        Field::new(
            "guide_id",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new(
            "target_gene",
            DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            true,
        ),
        Field::new("umi_count", DataType::UInt32, true),
        Field::new("assignment_confidence", DataType::Float32, true),
        Field::new(
            "assignment_model",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            false,
        ),
        Field::new("is_unassigned", DataType::Boolean, false),
        Field::new("is_multi_infected", DataType::Boolean, false),
        Field::new("n_guides_detected", DataType::UInt8, false),
    ])
}

pub fn assignment_schema_ref() -> Arc<Schema> {
    Arc::new(assignment_schema())
}

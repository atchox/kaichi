use arrow::record_batch::RecordBatch;
use anyhow::Result;
use serde_json::Value;

pub mod umi;
pub mod max;
pub mod ratio;
pub mod poisson_gauss;

/// Input to all assignment models: the sparse count matrix plus per-cell covariates,
/// both represented as Arrow RecordBatches.
///
/// `counts` schema: (cell_barcode: Utf8, guide_id: Utf8, umi_count: UInt32) — long form.
/// `covariates` schema: (cell_barcode: Utf8, batch: Utf8, total_counts: Float32).
pub struct AssignmentInput {
    pub counts: RecordBatch,
    pub covariates: RecordBatch,
}

pub trait AssignmentModel: Send + Sync {
    fn name(&self) -> &'static str;

    /// Run the model and return one row per cell in the assignment schema.
    fn assign(&self, input: &AssignmentInput) -> Result<RecordBatch>;

    /// Parameters as JSON, recorded in uns['kaichi']['model_params'].
    fn params_json(&self) -> Value;
}

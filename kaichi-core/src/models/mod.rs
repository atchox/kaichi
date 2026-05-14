use crate::data::{AssignmentResult, LoadedInput};
use anyhow::Result;
use serde_json::Value;

pub mod em;
pub mod gauss;
pub mod max;
pub mod output;
pub mod poisson_gauss;
pub mod ratio;
pub mod umi;

pub trait AssignmentModel: Send + Sync {
    fn name(&self) -> &'static str;

    /// Run the model and return an `AssignmentResult`.
    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult>;

    /// Parameters as JSON, recorded in uns['kaichi']['model_params'].
    fn params_json(&self) -> Value;
}

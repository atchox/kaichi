use crate::data::{AssignmentResult, LoadedInput};
use crate::score::ScoreMatrix;
use anyhow::Result;
use serde_json::Value;

pub mod beta;
pub mod binomial;
pub mod em;
pub mod em_count_mixture;
pub mod gauss;
pub mod max;
pub mod neg_binomial;
pub mod output;
pub mod poisson;
pub mod poisson_gauss;
pub mod quantiles;
pub mod ratio;
pub mod umi;

#[cfg(test)]
pub(crate) mod test_support;

pub trait AssignmentModel: Send + Sync {
    fn name(&self) -> &'static str;

    /// Run the model and return an `AssignmentResult`.
    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult>;

    /// Parameters as JSON, recorded in uns['kaichi']['model_params'].
    fn params_json(&self) -> Value;
}

/// Two-stage interface for models where score caching pays off.
///
/// All EM mixture models and `quantiles` implement this. Single-stage models
/// (`umi`, `ratio`, `max`) implement only `AssignmentModel`.
pub trait TwoStage: AssignmentModel {
    /// Produce the sparse score matrix (expensive: runs EM or equivalent).
    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix>;

    /// Apply a confidence threshold to a score matrix (cheap: one linear pass).
    fn decide(&self, scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult>;
}

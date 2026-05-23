//! Generic orchestration for count-based EM mixture models.
//!
//! `EmCountMixture` captures the per-model math (init, E+M cycle, posterior).
//! `score_em_count_mixture` is the single canonical per-guide fitting loop and
//! per-cell scoring loop, shared by `poisson`, `neg_binomial`, and `binomial`.
//! `decide_threshold` is the single canonical per-cell decision loop used by
//! all posterior-threshold models.

use crate::data::{AssignmentResult, LoadedInput};
use crate::models::em::run_em;
use crate::models::output::{n_detected_u8, AssignmentOutputBuilder};
use crate::score::ScoreMatrix;

use anyhow::Result;
use arrow::array::{Float32Array, UInt32Array};
use rayon::prelude::*;
use serde_json::Value;

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Gating thresholds: guides with too few nonzero cells or too low a max count
/// are skipped (no EM fit attempted).
pub struct Gating {
    pub min_nonzero: u32,
    pub min_max_count: u32,
}

/// EM algorithm configuration.
pub struct EmConfig {
    pub max_iters: u32,
    pub n_restarts: u32,
    pub tol: f64,
}

/// Nonzero entries for a single guide column, assembled by the model.
///
/// `triples[k] = (count_value, per_cell_covariate, batch_code)`. The covariate
/// interpretation is model-specific: log_depth for Poisson/NB, trial_count for
/// Binomial. `n_zeros` is used by models that include zero observations in the
/// E-step (currently only `poisson_gauss`, which is implemented separately).
pub struct GuideData {
    pub triples: Vec<(f64, f64, u16)>,
    pub n_zeros: usize,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Abstracts the model-specific math for count-based EM mixture models.
///
/// Implementors: `PoissonModel`, `NegBinomialModel`, `BinomialModel`.
/// `PoissonGaussModel` is handled separately because its data pipeline differs
/// substantially (zero-inclusive E-step, no per-cell depth covariate).
pub trait EmCountMixture: Send + Sync {
    type FitParams: Clone + Send;

    fn gating(&self) -> Gating;
    fn em_config(&self) -> EmConfig;

    /// Build the per-cell covariate array. Called once before the per-guide
    /// fitting loop. Default: `log(total_count + 1)` (depth normalization).
    fn prepare_covariates(&self, input: &LoadedInput) -> Vec<f64> {
        let tc = &input.covariates.total_counts;
        (0..input.counts.n_cells)
            .map(|i| (tc.value(i) as f64 + 1.0).ln())
            .collect()
    }

    /// Assemble guide data from a CSC column's nonzero rows. Default: filter
    /// `count > 0`, map to `(count, covariate[cell], batch[cell])`.
    fn guide_data(
        &self,
        col_row_indices: &[usize],
        col_values: &[u32],
        covariates: &[f64],
        batch_codes: &[u16],
        _n_cells: usize,
    ) -> GuideData {
        let triples = col_row_indices
            .iter()
            .zip(col_values)
            .filter(|(_, &v)| v > 0)
            .map(|(&i, &v)| (v as f64, covariates[i], batch_codes[i]))
            .collect();
        GuideData { triples, n_zeros: 0 }
    }

    /// Initialize params for one EM restart with the given mixing proportion π.
    fn init(&self, data: &GuideData, n_batches: usize, pi_k: f64) -> Self::FitParams;

    /// One full E→M cycle. `responsibilities` is pre-allocated to length
    /// `data.triples.len()` and reused across iterations (reset by caller).
    fn em_cycle(
        &self,
        params: Self::FitParams,
        data: &GuideData,
        n_batches: usize,
        responsibilities: &mut Vec<f64>,
    ) -> (Self::FitParams, f64);

    /// Signal posterior for a single (count, covariate, batch) observation.
    fn posterior(
        &self,
        count: u32,
        covariate: f64,
        params: &Self::FitParams,
        batch: u16,
    ) -> f32;
}

// ---------------------------------------------------------------------------
// Generic scoring pass
// ---------------------------------------------------------------------------

/// Fit per-guide EM models and build a sparse score matrix, for any model
/// implementing `EmCountMixture`.
pub fn score_em_count_mixture<M: EmCountMixture>(
    model: &M,
    model_kind: crate::score::ModelKind,
    model_params: Value,
    input: &LoadedInput,
) -> Result<ScoreMatrix> {
    let n_cells = input.counts.n_cells;
    let n_guides = input.counts.n_guides;
    let csc = input.counts.csc();
    let batch = &input.covariates.batch;
    let n_batches = batch.n_batches();
    let config = model.em_config();
    let gating = model.gating();
    let n_restarts = config.n_restarts.max(1);

    let covariates = model.prepare_covariates(input);

    // Parallel per-guide EM fitting.
    let guide_fits: Vec<Option<M::FitParams>> = (0..n_guides)
        .into_par_iter()
        .map(|g| {
            let col = csc.get_col(g).unwrap();
            let data = model.guide_data(
                col.row_indices(),
                col.values(),
                &covariates,
                &batch.codes,
                n_cells,
            );
            let n_nonzero = data.triples.len() as u32;
            let max_count = data
                .triples
                .iter()
                .map(|(y, _, _)| *y as u32)
                .max()
                .unwrap_or(0);
            if n_nonzero < gating.min_nonzero || max_count < gating.min_max_count {
                return None;
            }

            let mut responsibilities = vec![0.0f64; data.triples.len()];
            let mut best: Option<(M::FitParams, f64)> = None;

            for k in 0..n_restarts {
                let pi_k = (k as f64 + 1.0) / (n_restarts as f64 + 1.0) * 0.5;
                let init_params = model.init(&data, n_batches, pi_k);
                for r in responsibilities.iter_mut() {
                    *r = 0.0;
                }
                let (params, ll) = run_em(
                    init_params,
                    |p| model.em_cycle(p, &data, n_batches, &mut responsibilities),
                    config.max_iters,
                    config.tol,
                );
                if best.as_ref().map_or(true, |(_, bll)| ll > *bll) {
                    best = Some((params, ll));
                }
            }

            best.map(|(p, _)| p)
        })
        .collect();

    // Serial per-cell scoring pass — same traversal pattern as today's per-cell
    // loop in each model, now shared across all EmCountMixture implementors.
    let csr = input.counts.csr();
    let mut score_data: Vec<f32> = Vec::with_capacity(csr.nnz());
    let mut umi_data: Vec<u32> = Vec::with_capacity(csr.nnz());
    let mut score_indices: Vec<u32> = Vec::with_capacity(csr.nnz());
    let mut score_indptr: Vec<u32> = Vec::with_capacity(n_cells + 1);
    score_indptr.push(0);

    for cell in 0..n_cells {
        let row = csr.get_row(cell).unwrap();
        let covariate = covariates[cell];
        let b = batch.codes[cell];

        for (&guide_idx, &count) in row.col_indices().iter().zip(row.values()) {
            if let Some(params) = &guide_fits[guide_idx] {
                let score = model.posterior(count, covariate, params, b);
                score_data.push(score);
                umi_data.push(count);
                score_indices.push(guide_idx as u32);
            }
            // Guide gated out → entry omitted from score matrix.
        }
        score_indptr.push(score_data.len() as u32);
    }

    Ok(ScoreMatrix {
        data: Float32Array::from(score_data),
        umi_counts: UInt32Array::from(umi_data),
        indices: UInt32Array::from(score_indices),
        indptr: UInt32Array::from(score_indptr),
        cell_barcodes: input.covariates.cell_barcodes.clone(),
        guide_ids: input.guide_metadata.guide_ids.clone(),
        model: model_kind,
        model_params,
    })
}

// ---------------------------------------------------------------------------
// Generic decision pass
// ---------------------------------------------------------------------------

/// Apply a confidence threshold to a score matrix and produce an assignment
/// table. Used by all posterior-threshold two-stage models.
pub fn decide_threshold(scores: &ScoreMatrix, min_confidence: f32) -> Result<AssignmentResult> {
    let n_cells = scores.n_cells();
    let n_guides = scores.n_guides();
    let guide_ids = &scores.guide_ids;
    let model_name = scores.model.name();

    let mut out = AssignmentOutputBuilder::new(n_cells, n_guides, model_name);

    for cell in 0..n_cells {
        let mut best: Option<(u32, f32, u32)> = None; // (guide_idx, score, count)
        let mut n_passing: usize = 0;

        for (guide_idx, score, count) in scores.row_with_counts(cell) {
            if score >= min_confidence {
                n_passing += 1;
                match best {
                    None => best = Some((guide_idx, score, count)),
                    Some((_, bs, bc)) => {
                        if score > bs || (score == bs && count > bc) {
                            best = Some((guide_idx, score, count));
                        }
                    }
                }
                out.push_assigned_triple(cell, guide_idx as usize);
            }
        }

        let n_det = n_detected_u8(n_passing);
        match best {
            None => out.append_unassigned(n_det),
            Some((guide_idx, score, count)) => {
                out.append_assigned(
                    guide_ids.value(guide_idx as usize),
                    count,
                    score,
                    n_passing > 1,
                    n_det,
                );
            }
        }
    }

    out.finish(&scores.cell_barcodes, true)
}

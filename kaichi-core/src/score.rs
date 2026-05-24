use arrow::array::{Array as _, Float32Array, StringArray, UInt32Array};
use serde_json::Value;

/// Identifies which model produced a `ScoreMatrix`.
#[derive(Clone, Debug)]
pub enum ModelKind {
    PoissonGauss,
    Poisson,
    NegBinomial,
    Binomial,
    Beta2,
    Beta3,
    Quantiles,
    Umi,
    Ratio,
    Max,
}

impl ModelKind {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PoissonGauss => "poisson_gauss",
            Self::Poisson => "poisson",
            Self::NegBinomial => "neg_binomial",
            Self::Binomial => "binomial",
            Self::Beta2 => "beta2",
            Self::Beta3 => "beta3",
            Self::Quantiles => "quantiles",
            Self::Umi => "umi",
            Self::Ratio => "ratio",
            Self::Max => "max",
        }
    }

    /// Inverse of `name()`. Used when reconstructing a `ScoreMatrix` from disk.
    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "poisson_gauss" => Self::PoissonGauss,
            "poisson"       => Self::Poisson,
            "neg_binomial"  => Self::NegBinomial,
            "binomial"      => Self::Binomial,
            "beta2"         => Self::Beta2,
            "beta3"         => Self::Beta3,
            "quantiles"     => Self::Quantiles,
            "umi"           => Self::Umi,
            "ratio"         => Self::Ratio,
            "max"           => Self::Max,
            _ => return None,
        })
    }
}

/// Sparse CSR score matrix produced by the scoring stage.
///
/// Sparsity pattern matches the input counts: only `(cell, guide)` pairs with
/// `count > 0` carry a score and a UMI count. The three CSR arrays follow the
/// Arrow convention used throughout the codebase.
pub struct ScoreMatrix {
    /// Signal posterior (or equivalent score) for each nonzero entry.
    pub data: Float32Array,
    /// UMI count for each nonzero entry (parallel to `data`). Required by `decide`.
    pub umi_counts: UInt32Array,
    /// Guide index for each nonzero entry (column indices in CSR).
    pub indices: UInt32Array,
    /// Row start positions, length `n_cells + 1`.
    pub indptr: UInt32Array,

    pub cell_barcodes: StringArray,
    pub guide_ids: StringArray,

    pub model: ModelKind,
    /// Score-time params only (no decision thresholds), stored for provenance.
    pub model_params: Value,
}

impl ScoreMatrix {
    pub fn n_cells(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    pub fn n_guides(&self) -> usize {
        self.guide_ids.len()
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Iterate `(guide_idx, score)` pairs for one cell. Zero allocation.
    pub fn row(&self, cell: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let lo = self.indptr.value(cell) as usize;
        let hi = self.indptr.value(cell + 1) as usize;
        (lo..hi).map(|k| (self.indices.value(k), self.data.value(k)))
    }

    /// Iterate `(guide_idx, score, umi_count)` for one cell. Zero allocation.
    pub fn row_with_counts(&self, cell: usize) -> impl Iterator<Item = (u32, f32, u32)> + '_ {
        let lo = self.indptr.value(cell) as usize;
        let hi = self.indptr.value(cell + 1) as usize;
        (lo..hi).map(|k| (self.indices.value(k), self.data.value(k), self.umi_counts.value(k)))
    }
}

use anyhow::{anyhow, Result};
use arrow::{
    array::{Float32Array, StringArray},
    record_batch::RecordBatch,
};
use nalgebra_sparse::{CscMatrix, CsrMatrix};
use std::sync::OnceLock;

/// Sparse cells × guides UMI count matrix.
///
/// Owns the CSR matrix. CSC is computed once on first access and cached.
pub struct CountMatrix {
    csr: CsrMatrix<u32>,
    csc: OnceLock<CscMatrix<u32>>,
    pub n_cells: usize,
    pub n_guides: usize,
}

impl CountMatrix {
    pub fn try_from_csr(
        n_cells: usize,
        n_guides: usize,
        row_offsets: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<u32>,
    ) -> Result<Self> {
        let csr = CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, values)
            .map_err(|e| anyhow!("CSR construction failed: {e:?}"))?;
        Ok(Self { csr, csc: OnceLock::new(), n_cells, n_guides })
    }

    pub fn csr(&self) -> &CsrMatrix<u32> {
        &self.csr
    }

    /// CSC view, built once on first call via an O(nnz) transpose.
    pub fn csc(&self) -> &CscMatrix<u32> {
        self.csc.get_or_init(|| CscMatrix::from(&self.csr))
    }

    pub fn nnz(&self) -> usize {
        self.csr.nnz()
    }
}

/// Per-cell metadata: barcodes and guide-count totals.
pub struct Covariates {
    pub cell_barcodes: StringArray,
    pub total_counts: Float32Array,
}

/// Per-guide metadata: guide IDs and optional library annotations.
pub struct GuideMetadata {
    pub guide_ids: StringArray,
}

/// Everything produced by the IO layer, passed to assignment models.
pub struct LoadedInput {
    pub counts: CountMatrix,
    pub covariates: Covariates,
    pub guide_metadata: GuideMetadata,
}

/// Output of an assignment model: per-cell RecordBatch + binary assignment layer.
pub struct AssignmentResult {
    /// Schema defined in schema.rs; one row per cell.
    pub batch: RecordBatch,
    /// Binary cells × guides CSR (1 = assigned).
    pub assigned_x: CsrMatrix<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_csr() -> CountMatrix {
        // 3 cells, 2 guides
        // cell 0: guide 1 = 5
        // cell 1: guide 0 = 3, guide 1 = 1
        // cell 2: (empty)
        CountMatrix::try_from_csr(
            3, 2,
            vec![0, 1, 3, 3],      // row_offsets
            vec![1, 0, 1],         // col_indices
            vec![5, 3, 1],         // values
        )
        .unwrap()
    }

    #[test]
    fn try_from_csr_round_trips() {
        let m = tiny_csr();
        assert_eq!(m.n_cells, 3);
        assert_eq!(m.n_guides, 2);
        assert_eq!(m.nnz(), 3);
    }

    #[test]
    fn csr_row_access_correct() {
        let m = tiny_csr();
        let row0 = m.csr().get_row(0).unwrap();
        assert_eq!(row0.col_indices(), &[1]);
        assert_eq!(row0.values(), &[5]);

        let row1 = m.csr().get_row(1).unwrap();
        assert_eq!(row1.col_indices(), &[0, 1]);
        assert_eq!(row1.values(), &[3, 1]);

        let row2 = m.csr().get_row(2).unwrap();
        assert_eq!(row2.nnz(), 0);
    }

    #[test]
    fn csc_lazy_init_correct() {
        let m = tiny_csr();
        // guide 0: cell 1 = 3
        let col0 = m.csc().get_col(0).unwrap();
        assert_eq!(col0.row_indices(), &[1]);
        assert_eq!(col0.values(), &[3]);

        // guide 1: cell 0 = 5, cell 1 = 1
        let col1 = m.csc().get_col(1).unwrap();
        assert_eq!(col1.row_indices(), &[0, 1]);
        assert_eq!(col1.values(), &[5, 1]);
    }

    #[test]
    fn csc_called_twice_returns_same_instance() {
        let m = tiny_csr();
        let ptr_a = m.csc() as *const _;
        let ptr_b = m.csc() as *const _;
        assert_eq!(ptr_a, ptr_b, "OnceLock must return same allocation");
    }

    #[test]
    fn empty_matrix_ok() {
        let m = CountMatrix::try_from_csr(0, 0, vec![0], vec![], vec![]).unwrap();
        assert_eq!(m.nnz(), 0);
        assert_eq!(m.csc().nnz(), 0);
    }

    #[test]
    fn single_cell_single_guide() {
        let m = CountMatrix::try_from_csr(1, 1, vec![0, 1], vec![0], vec![42]).unwrap();
        assert_eq!(m.csr().get_row(0).unwrap().values(), &[42]);
        assert_eq!(m.csc().get_col(0).unwrap().values(), &[42]);
    }

    #[test]
    fn csc_nnz_matches_csr_nnz() {
        let m = tiny_csr();
        assert_eq!(m.csr().nnz(), m.csc().nnz());
    }

    #[test]
    fn column_sum_equals_row_sum() {
        let m = tiny_csr();
        let row_total: u32 = (0..m.n_cells)
            .map(|r| m.csr().get_row(r).unwrap().values().iter().sum::<u32>())
            .sum();
        let col_total: u32 = (0..m.n_guides)
            .map(|c| m.csc().get_col(c).unwrap().values().iter().sum::<u32>())
            .sum();
        assert_eq!(row_total, col_total);
    }

    #[test]
    fn invalid_csr_data_returns_err() {
        // col index 5 is out of range for 2 guides
        let result = CountMatrix::try_from_csr(2, 2, vec![0, 1, 2], vec![0, 5], vec![1, 1]);
        assert!(result.is_err());
    }
}

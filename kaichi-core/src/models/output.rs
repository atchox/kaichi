//! Shared output construction for assignment models.
//!
//! Every model produces the same Arrow schema and `CsrMatrix<u8>` assignment
//! layer. The boilerplate is identical: seven column builders, a triples vec
//! for the sparse layer, then a final pass to build the RecordBatch.
//!
//! `AssignmentOutputBuilder` owns those builders and exposes two append paths
//! (`append_unassigned`, `append_assigned`) plus a `finish` that produces the
//! `AssignmentResult`. All hot-path methods are `#[inline]` so call sites
//! compile to the same code as the hand-rolled per-model versions.
//!
//! `assigned_x` semantics differ by model:
//! - Single-guide assignment (`umi` for 0/1 above-threshold cases, `max`,
//!   `ratio`, `gauss`, `poisson_gauss` single-assigned): push exactly
//!   `(cell, guide)` for the assigned guide.
//! - Multi-infected (`umi` for 2+, `poisson_gauss` multi): push one
//!   `(cell, guide)` per passing guide via `push_assigned_triple`.

use crate::data::AssignmentResult;
use crate::schema::assignment_schema_ref;

use anyhow::{Context, Result};
use arrow::{
    array::{
        BooleanBuilder, DictionaryArray, Float32Builder, StringArray,
        StringDictionaryBuilder, UInt32Builder, UInt8Builder,
    },
    datatypes::Int16Type,
    record_batch::RecordBatch,
};
use nalgebra_sparse::CsrMatrix;
use std::sync::Arc;

pub struct AssignmentOutputBuilder {
    pub n_cells: usize,
    pub n_guides: usize,
    pub model_name: &'static str,

    guide_ids: StringDictionaryBuilder<Int16Type>,
    target_genes: StringDictionaryBuilder<Int16Type>,
    umi_counts: UInt32Builder,
    confidence: Float32Builder,
    is_unassigned: BooleanBuilder,
    is_multi: BooleanBuilder,
    n_detected: UInt8Builder,

    /// Sorted list of `(cell_idx, guide_idx)` pairs for `assigned_x`.
    /// Models that already emit pairs in cell-major order can skip the final sort.
    assigned_triples: Vec<(usize, usize)>,
}

impl AssignmentOutputBuilder {
    pub fn new(n_cells: usize, n_guides: usize, model_name: &'static str) -> Self {
        Self {
            n_cells,
            n_guides,
            model_name,
            guide_ids: StringDictionaryBuilder::new(),
            target_genes: StringDictionaryBuilder::new(),
            umi_counts: UInt32Builder::with_capacity(n_cells),
            confidence: Float32Builder::with_capacity(n_cells),
            is_unassigned: BooleanBuilder::with_capacity(n_cells),
            is_multi: BooleanBuilder::with_capacity(n_cells),
            n_detected: UInt8Builder::with_capacity(n_cells),
            assigned_triples: Vec::new(),
        }
    }

    /// Append a row for an unassigned cell. `n_detected` is the number of
    /// guides that passed model-specific gating, even though none were assigned.
    #[inline]
    pub fn append_unassigned(&mut self, n_detected: u8) {
        self.guide_ids.append_null();
        self.target_genes.append_null();
        self.umi_counts.append_null();
        self.confidence.append_null();
        self.is_unassigned.append_value(true);
        self.is_multi.append_value(false);
        self.n_detected.append_value(n_detected);
    }

    /// Append a row for an assigned cell. **Does not push to `assigned_x`** —
    /// the caller must invoke [`push_assigned_triple`] explicitly. This split
    /// keeps the API uniform across models that record one triple per cell
    /// (`max`, `ratio`) and models that record every passing guide
    /// (`umi` multi-infected, `poisson_gauss`).
    #[inline]
    pub fn append_assigned(
        &mut self,
        guide_name: &str,
        umi_count: u32,
        confidence: f32,
        is_multi: bool,
        n_detected: u8,
    ) {
        self.guide_ids.append_value(guide_name);
        self.target_genes.append_null();
        self.umi_counts.append_value(umi_count);
        self.confidence.append_value(confidence);
        self.is_unassigned.append_value(false);
        self.is_multi.append_value(is_multi);
        self.n_detected.append_value(n_detected);
    }

    /// Push a `(cell, guide)` pair into the assigned-layer triples.
    #[inline]
    pub fn push_assigned_triple(&mut self, cell_idx: usize, guide_idx: usize) {
        self.assigned_triples.push((cell_idx, guide_idx));
    }

    /// Finalize the builder into an `AssignmentResult`.
    ///
    /// `triples_sorted`: if true, skip the sort+dedup step (the caller guarantees
    /// the triples are already strictly increasing in `(cell, guide)` order).
    /// Set this to true when triples are emitted in a single cell-major loop
    /// with at most one push per `(cell, guide)`.
    pub fn finish(
        mut self,
        cell_barcodes: &StringArray,
        triples_sorted: bool,
    ) -> Result<AssignmentResult> {
        let model_name_arr: DictionaryArray<arrow::datatypes::Int8Type> = {
            let values = StringArray::from(vec![self.model_name]);
            let keys = arrow::array::Int8Array::from(vec![0i8; self.n_cells]);
            DictionaryArray::try_new(keys, Arc::new(values))
                .context("failed to build assignment_model dictionary")?
        };

        let batch = RecordBatch::try_new(
            assignment_schema_ref(),
            vec![
                Arc::new(cell_barcodes.clone()),
                Arc::new(self.guide_ids.finish()),
                Arc::new(self.target_genes.finish()),
                Arc::new(self.umi_counts.finish()),
                Arc::new(self.confidence.finish()),
                Arc::new(model_name_arr),
                Arc::new(self.is_unassigned.finish()),
                Arc::new(self.is_multi.finish()),
                Arc::new(self.n_detected.finish()),
            ],
        )
        .context("failed to build output RecordBatch")?;

        if !triples_sorted {
            self.assigned_triples.sort_unstable();
            self.assigned_triples.dedup();
        }
        let assigned_x = build_assigned_csr(self.assigned_triples, self.n_cells, self.n_guides);

        Ok(AssignmentResult { batch, assigned_x })
    }
}

/// Build a CSR<u8> from sorted, deduplicated `(cell, guide)` triples.
/// All values are 1; the caller must guarantee that triples are sorted
/// (strictly increasing in `(cell, guide)` order) before calling.
pub fn build_assigned_csr(
    sorted_triples: Vec<(usize, usize)>,
    n_cells: usize,
    n_guides: usize,
) -> CsrMatrix<u8> {
    let nnz = sorted_triples.len();
    let mut row_offsets = vec![0usize; n_cells + 1];
    let mut col_indices = Vec::with_capacity(nnz);

    let mut last_row = 0usize;
    for (idx, &(r, c)) in sorted_triples.iter().enumerate() {
        while last_row < r {
            row_offsets[last_row + 1] = idx;
            last_row += 1;
        }
        col_indices.push(c);
    }
    for i in (last_row + 1)..=n_cells {
        row_offsets[i] = nnz;
    }

    CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, vec![1u8; nnz])
        .expect("assigned CSR from sorted unique triples cannot fail")
}

/// Saturating cast for `n_detected` column (u8 storage, but counts can in
/// principle exceed 255 in pathological libraries).
#[inline]
pub fn n_detected_u8(n: usize) -> u8 {
    n.min(255) as u8
}

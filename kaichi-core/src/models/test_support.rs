//! Shared fixtures for model unit tests. Test-only.
//!
//! `make_input` / `make_input_with_totals` were duplicated across every model's
//! `mod tests` — same CSR plumbing each time. This module owns the construction
//! once so each test site only declares the parts that matter.

#![cfg(test)]

use crate::data::{BatchLabels, CountMatrix, Covariates, GuideMetadata, LoadedInput};
use arrow::array::{Float32Array, StringBuilder};

/// Build a `LoadedInput` from sparse `(row, col, value)` triples.
/// `total_counts` is supplied by the caller (one Float32 per cell).
/// Cell barcodes are `C0..C{n_cells-1}`; guide IDs are `g0..g{n_guides-1}`.
pub fn input_with_totals(
    n_cells: usize,
    n_guides: usize,
    triples: Vec<(usize, usize, u32)>,
    totals: Vec<f32>,
) -> LoadedInput {
    assert_eq!(totals.len(), n_cells, "totals length must equal n_cells");

    let mut sorted = triples;
    sorted.sort_unstable_by_key(|&(r, c, _)| (r, c));
    let nnz = sorted.len();
    let mut row_offsets = vec![0usize; n_cells + 1];
    let mut col_indices = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);
    let mut last = 0usize;
    for (idx, &(r, c, v)) in sorted.iter().enumerate() {
        while last < r {
            row_offsets[last + 1] = idx;
            last += 1;
        }
        col_indices.push(c);
        values.push(v);
    }
    for i in (last + 1)..=n_cells {
        row_offsets[i] = nnz;
    }

    let counts = CountMatrix::try_from_csr(n_cells, n_guides, row_offsets, col_indices, values)
        .expect("CSR construction failed");

    let mut bc = StringBuilder::new();
    for i in 0..n_cells {
        bc.append_value(format!("C{i}"));
    }
    let mut gd = StringBuilder::new();
    for i in 0..n_guides {
        gd.append_value(format!("g{i}"));
    }

    LoadedInput {
        counts,
        covariates: Covariates {
            cell_barcodes: bc.finish(),
            total_counts: Float32Array::from(totals),
            batch: BatchLabels::single_batch(n_cells),
        },
        guide_metadata: GuideMetadata { guide_ids: gd.finish() },
    }
}

/// `total_counts` derived from row sums of the count matrix.
/// Use this when the model treats `total_counts` as "sum of guide UMIs per cell".
pub fn input_with_row_sums(
    n_cells: usize,
    n_guides: usize,
    triples: Vec<(usize, usize, u32)>,
) -> LoadedInput {
    let mut totals = vec![0u32; n_cells];
    for &(r, _, v) in &triples {
        totals[r] += v;
    }
    input_with_totals(
        n_cells,
        n_guides,
        triples,
        totals.iter().map(|&t| t as f32).collect(),
    )
}

/// `total_counts` is all zeros — for models that don't use it as a covariate
/// (e.g. `umi`, `max`).
pub fn input_zero_totals(
    n_cells: usize,
    n_guides: usize,
    triples: Vec<(usize, usize, u32)>,
) -> LoadedInput {
    input_with_totals(n_cells, n_guides, triples, vec![0.0; n_cells])
}

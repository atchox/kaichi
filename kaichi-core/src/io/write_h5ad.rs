use crate::models::AssignmentInput;
use anndata::{
    AnnDataOp, AxisArraysOp,
    backend::{WriteConfig, Compression},
    data::{DataFrameIndex, DynCsrMatrix, ArrayData},
};
use anndata_hdf5::H5;
use anyhow::{Context, Result};
use arrow::{
    array::{Array, BooleanArray, StringArray, UInt8Array, UInt32Array},
    compute::cast,
    datatypes::DataType,
    record_batch::RecordBatch,
};
use nalgebra_sparse::CsrMatrix;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Write a kaichi assignment as a standalone H5AD.
///
/// Layout follows data-model.md and matches the crispat module's output shape:
///   X                            sparse UInt32 (cells × guides) — raw UMI counts
///   layers["assigned"]           sparse UInt8  (cells × guides) — binary call
///   obs.guide_identity           str — assigned guide ID, "" if unassigned
///   obs.n_guides_assigned        i32 — from result.n_guides_detected
///   var.index                    guide IDs
///   uns.guide_assignment_method  "kaichi_{model_name}"
pub fn write_h5ad(
    input: &AssignmentInput,
    result: &RecordBatch,
    obs_names: &[String],
    var_names: &[String],
    path: &Path,
    model_name: &str,
) -> Result<()> {
    // Write with gzip so h5py can read it without external HDF5 filter plugins.
    // The anndata-hdf5 default is blosc_zstd (filter 32001), which needs
    // hdf5-external-filter-plugins in the conda env.
    anndata::backend::set_default_write_config(WriteConfig {
        compression: Some(Compression::Gzip(3)),
        ..Default::default()
    });

    let n_cells = obs_names.len();
    let n_guides = var_names.len();

    let cell_idx: HashMap<&str, usize> = obs_names
        .iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
    let guide_idx: HashMap<&str, usize> = var_names
        .iter().enumerate().map(|(i, g)| (g.as_str(), i)).collect();

    let x_csr = build_x_csr(&input.counts, &cell_idx, &guide_idx, n_cells, n_guides)?;
    let (assigned_csr, guide_identity, n_guides_assigned) =
        build_assigned(result, &cell_idx, &guide_idx, n_cells, n_guides)?;

    // obs DataFrame
    let cols = vec![
        Series::new("guide_identity".into(), guide_identity).into_column(),
        Series::new("n_guides_assigned".into(), n_guides_assigned).into_column(),
    ];
    let df = DataFrame::new(n_cells, cols).context("failed to build obs DataFrame")?;

    // Create and populate AnnData
    let adata = anndata::AnnData::<H5>::new(path)
        .context("failed to create H5AD file")?;

    adata.set_obs_names(obs_names.iter().cloned().collect::<DataFrameIndex>())
        .context("failed to set obs names")?;
    adata.set_var_names(var_names.iter().cloned().collect::<DataFrameIndex>())
        .context("failed to set var names")?;
    adata.set_x(ArrayData::CsrMatrix(DynCsrMatrix::U32(x_csr)))
        .context("failed to set X")?;
    adata.set_obs(df).context("failed to set obs")?;
    adata.layers().add("assigned", ArrayData::CsrMatrix(DynCsrMatrix::U8(assigned_csr)))
        .context("failed to add assigned layer")?;
    adata.set_uns(vec![
        ("guide_assignment_method".to_string(), format!("kaichi_{model_name}")),
    ])
    .context("failed to set uns")?;

    Ok(())
}

/// Build the cells × guides UMI count matrix in CSR uint32 form from the long-form counts batch.
fn build_x_csr(
    counts: &RecordBatch,
    cell_idx: &HashMap<&str, usize>,
    guide_idx: &HashMap<&str, usize>,
    n_cells: usize,
    n_guides: usize,
) -> Result<CsrMatrix<u32>> {
    let barcodes = counts.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let guide_ids = counts.column_by_name("guide_id").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let umi = counts.column_by_name("umi_count").unwrap()
        .as_any().downcast_ref::<UInt32Array>().unwrap();

    let mut triples: Vec<(usize, usize, u32)> = Vec::with_capacity(counts.num_rows());
    for i in 0..counts.num_rows() {
        let v = umi.value(i);
        if v == 0 { continue; }
        let Some(&ci) = cell_idx.get(barcodes.value(i)) else { continue };
        let Some(&gi) = guide_idx.get(guide_ids.value(i)) else { continue };
        triples.push((ci, gi, v));
    }
    triples.sort_unstable_by_key(|&(r, c, _)| (r, c));

    let nnz = triples.len();
    let col_indices: Vec<usize> = triples.iter().map(|&(_, c, _)| c).collect();
    let values: Vec<u32> = triples.iter().map(|&(_, _, v)| v).collect();
    let row_offsets = build_row_offsets(triples.iter().map(|&(r, _, _)| r), n_cells, nnz);

    CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, values)
        .map_err(|e| anyhow::anyhow!("failed to build X CSR matrix: {}", e))
}

/// Build the binary cells × guides assignment matrix plus the obs columns from result.
fn build_assigned(
    result: &RecordBatch,
    cell_idx: &HashMap<&str, usize>,
    guide_idx: &HashMap<&str, usize>,
    n_cells: usize,
    n_guides: usize,
) -> Result<(CsrMatrix<u8>, Vec<String>, Vec<i32>)> {
    let barcodes = result.column_by_name("cell_barcode").unwrap()
        .as_any().downcast_ref::<StringArray>().unwrap();
    let guide_col = cast(result.column_by_name("guide_id").unwrap().as_ref(), &DataType::Utf8)?;
    let guide_ids = guide_col.as_any().downcast_ref::<StringArray>().unwrap();
    let is_unassigned = result.column_by_name("is_unassigned").unwrap()
        .as_any().downcast_ref::<BooleanArray>().unwrap();
    let n_detected = result.column_by_name("n_guides_detected").unwrap()
        .as_any().downcast_ref::<UInt8Array>().unwrap();

    let mut guide_identity = vec![String::new(); n_cells];
    let mut n_guides_assigned = vec![0i32; n_cells];
    let mut coo: Vec<(usize, usize)> = Vec::new();

    for i in 0..result.num_rows() {
        if is_unassigned.value(i) { continue; }
        let Some(&ci) = cell_idx.get(barcodes.value(i)) else { continue };
        let Some(&gi) = guide_idx.get(guide_ids.value(i)) else { continue };
        guide_identity[ci] = guide_ids.value(i).to_string();
        n_guides_assigned[ci] = n_detected.value(i) as i32;
        coo.push((ci, gi));
    }
    coo.sort_unstable();
    coo.dedup();

    let nnz = coo.len();
    let col_indices: Vec<usize> = coo.iter().map(|&(_, c)| c).collect();
    let values = vec![1u8; nnz];
    let row_offsets = build_row_offsets(coo.iter().map(|&(r, _)| r), n_cells, nnz);

    let csr = CsrMatrix::try_from_csr_data(n_cells, n_guides, row_offsets, col_indices, values)
        .map_err(|e| anyhow::anyhow!("failed to build assignment CSR matrix: {}", e))?;
    Ok((csr, guide_identity, n_guides_assigned))
}

/// Build CSR row offsets from a sorted iterator of row indices.
fn build_row_offsets<I: Iterator<Item = usize>>(rows: I, n_rows: usize, nnz: usize) -> Vec<usize> {
    let mut row_offsets = vec![0usize; n_rows + 1];
    let mut last = 0usize;
    for (idx, r) in rows.enumerate() {
        while last < r {
            row_offsets[last + 1] = idx;
            last += 1;
        }
    }
    for i in (last + 1)..=n_rows {
        row_offsets[i] = nnz;
    }
    row_offsets
}

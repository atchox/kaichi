use crate::models::AssignmentInput;

use anndata::{AnnDataOp, ArrayElemOp, data::DynCsrMatrix, ArrayData};
use anndata::backend::Backend;
use anndata_hdf5::H5;

use arrow::{
    array::{Float32Array, StringArray, UInt32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use anyhow::{bail, Context, Result};
use std::{path::Path, sync::Arc};

/// Read a guide-count H5AD and return an AssignmentInput ready for any model.
///
/// Expects:
///   obs_names — cell barcodes
///   var_names — guide IDs
///   X         — sparse UMI count matrix (cells × guides), any integer dtype
pub fn read_h5ad(path: &Path) -> Result<AssignmentInput> {
    let store = H5::open(path)
        .with_context(|| format!("cannot open H5AD: {}", path.display()))?;
    let adata = anndata::AnnData::<H5>::open(store)
        .context("cannot parse H5AD structure")?;

    let obs_names: Vec<String> = adata.obs_names().into_vec();
    let var_names: Vec<String> = adata.var_names().into_vec();
    let n_cells = obs_names.len();

    let x: ArrayData = adata.x().get::<ArrayData>()
        .context("cannot read X from H5AD")?
        .context("X is empty")?;

    let csr_u32 = match x {
        ArrayData::CsrMatrix(dyn_csr) => dyn_csr_to_u32(dyn_csr)?,
        ArrayData::CscMatrix(_) => bail!("CSC matrix input not supported; convert to CSR first"),
        other => bail!("unexpected X encoding: {:?}", other),
    };

    // Expand to long-form (cell_barcode, guide_id, umi_count).
    let mut barcodes: Vec<&str> = Vec::new();
    let mut guides: Vec<&str> = Vec::new();
    let mut counts: Vec<u32> = Vec::new();

    for (row_idx, row) in csr_u32.row_iter().enumerate() {
        let barcode = obs_names[row_idx].as_str();
        for (&col_idx, &val) in row.col_indices().iter().zip(row.values().iter()) {
            if val > 0 {
                barcodes.push(barcode);
                guides.push(var_names[col_idx].as_str());
                counts.push(val);
            }
        }
    }

    let counts_schema = Arc::new(Schema::new(vec![
        Field::new("cell_barcode", DataType::Utf8, false),
        Field::new("guide_id", DataType::Utf8, false),
        Field::new("umi_count", DataType::UInt32, false),
    ]));

    let counts_batch = RecordBatch::try_new(
        counts_schema,
        vec![
            Arc::new(StringArray::from(barcodes)),
            Arc::new(StringArray::from(guides)),
            Arc::new(UInt32Array::from(counts)),
        ],
    )
    .context("failed to build counts RecordBatch")?;

    // total_counts = per-cell UMI sum; batch = single default category.
    let total_counts: Vec<f32> = csr_u32
        .row_iter()
        .map(|row| row.values().iter().map(|&v| v as f32).sum())
        .collect();

    let covariate_schema = Arc::new(Schema::new(vec![
        Field::new("cell_barcode", DataType::Utf8, false),
        Field::new("batch", DataType::Utf8, false),
        Field::new("total_counts", DataType::Float32, false),
    ]));

    let covariate_batch = RecordBatch::try_new(
        covariate_schema,
        vec![
            Arc::new(StringArray::from(
                obs_names.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(vec!["default"; n_cells])),
            Arc::new(Float32Array::from(total_counts)),
        ],
    )
    .context("failed to build covariates RecordBatch")?;

    Ok(AssignmentInput {
        counts: counts_batch,
        covariates: covariate_batch,
    })
}

fn dyn_csr_to_u32(
    dyn_csr: DynCsrMatrix,
) -> Result<nalgebra_sparse::csr::CsrMatrix<u32>> {
    use DynCsrMatrix::*;
    macro_rules! cast_csr {
        ($mat:expr) => {{
            let (nrows, ncols) = ($mat.nrows(), $mat.ncols());
            let (offsets, indices, values) = $mat.disassemble();
            let values_u32: Vec<u32> = values.into_iter().map(|v| v as u32).collect();
            nalgebra_sparse::csr::CsrMatrix::try_from_csr_data(
                nrows, ncols, offsets, indices, values_u32,
            )
            .map_err(|e| anyhow::anyhow!("CSR construction failed: {:?}", e))
        }};
    }
    match dyn_csr {
        U32(m) => Ok(m),
        U8(m)  => cast_csr!(m),
        U16(m) => cast_csr!(m),
        U64(m) => cast_csr!(m),
        I8(m)  => cast_csr!(m),
        I16(m) => cast_csr!(m),
        I32(m) => cast_csr!(m),
        I64(m) => cast_csr!(m),
        F32(m) => cast_csr!(m),
        F64(m) => cast_csr!(m),
        _      => bail!("unsupported CsrMatrix element type for guide counts"),
    }
}

//! Python bindings for kaichi.
//!
//! This crate exposes a single private `#[pyfunction]` called from the
//! Python wrapper at `kaichi-py/python/kaichi/__init__.py`. See
//! `docs/design/binding-interop.md` ("Python v0.1") for the contract.

use kaichi_core::data::{AssignmentResult, LoadedInput};
use kaichi_core::io::read::read_h5ad;
use kaichi_core::models::{
    beta::{Beta2Model, Beta3Model},
    binomial::BinomialModel,
    gauss::GaussModel,
    max::MaxModel,
    neg_binomial::NegBinomialModel,
    poisson::PoissonModel,
    poisson_gauss::PoissonGaussModel,
    quantiles::QuantilesModel,
    ratio::RatioModel,
    umi::UmiModel,
    AssignmentModel,
};

use arrow::array::Array as _;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;
use std::path::Path;

/// Read an h5ad, run the named model, and return the pieces Python needs to
/// build an `anndata.AnnData` in memory.
///
/// Returns a 10-tuple:
///   (per_cell_batch, assigned_indptr, assigned_indices,
///    x_data, x_indices, x_indptr,
///    cell_barcodes, guide_ids,
///    model_name, model_params_json)
///
/// Per-cell columns travel as a `pyarrow.RecordBatch` (zero-copy via the Arrow
/// C Data Interface). CSR layouts travel as `numpy.uint32` arrays — the
/// binary `assigned` matrix's `data` array is trivially all-ones and is
/// reconstructed Python-side rather than shuttled across.
#[pyfunction]
#[pyo3(signature = (h5ad_path, model, *, min_confidence=None, quantile=None))]
fn _assign_from_h5ad_inmem<'py>(
    py: Python<'py>,
    h5ad_path: &str,
    model: &str,
    min_confidence: Option<f32>,
    quantile: Option<f32>,
) -> PyResult<(
    PyObject,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Vec<String>,
    Vec<String>,
    &'static str,
    String,
)> {
    let input = read_h5ad(Path::new(h5ad_path))
        .map_err(|e| PyIOError::new_err(format!("read_h5ad({h5ad_path}): {e:#}")))?;

    let (result, model_name, model_params_json) = run_model(model, &input, min_confidence, quantile)?;

    extract(py, input, result, model_name, model_params_json)
}

/// Dispatch on model name. Returns the assignment result, the model's static
/// name, and the model's params as a JSON string (for `uns["kaichi"]`).
fn run_model(
    name: &str,
    input: &LoadedInput,
    min_confidence: Option<f32>,
    quantile: Option<f32>,
) -> PyResult<(AssignmentResult, &'static str, String)> {
    macro_rules! run {
        ($model_expr:expr) => {{
            let m = $model_expr;
            let res = m.assign(input).map_err(map_assign_err)?;
            let params = serde_json::to_string(&m.params_json())
                .map_err(|e| PyValueError::new_err(format!("params_json: {e}")))?;
            (res, m.name(), params)
        }};
    }

    let out = match name {
        "umi"           => run!(UmiModel::default()),
        "max"           => run!(MaxModel::default()),
        "ratio"         => run!(RatioModel::default()),
        "gauss"         => {
            let mut m = GaussModel::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "poisson_gauss" => {
            let mut m = PoissonGaussModel::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "poisson"       => {
            let mut m = PoissonModel::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "neg_binomial"  => {
            let mut m = NegBinomialModel::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "binomial"      => {
            let mut m = BinomialModel::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "beta2"         => {
            let mut m = Beta2Model::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "beta3"         => {
            let mut m = Beta3Model::default();
            if let Some(c) = min_confidence { m.min_confidence = c; }
            run!(m)
        }
        "quantiles"     => run!(QuantilesModel {
            quantile: quantile.unwrap_or(0.1),
        }),
        other => return Err(PyValueError::new_err(format!(
            "unknown model {other:?}: expected one of umi/max/ratio/gauss/\
             poisson_gauss/poisson/neg_binomial/binomial/beta2/beta3/quantiles"
        ))),
    };
    Ok(out)
}

fn map_assign_err(e: anyhow::Error) -> PyErr {
    PyValueError::new_err(format!("model.assign failed: {e:#}"))
}

/// Pull the RecordBatch and the two CSR layouts out into Python-owned values.
fn extract<'py>(
    py: Python<'py>,
    input: LoadedInput,
    result: AssignmentResult,
    model_name: &'static str,
    model_params_json: String,
) -> PyResult<(
    PyObject,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Vec<String>,
    Vec<String>,
    &'static str,
    String,
)> {
    // assigned binary CSR: row_offsets + col_indices; data is implicit 1s.
    let assigned = &result.assigned_x;
    let a_indptr: Vec<u32> = assigned.row_offsets().iter().map(|&i| i as u32).collect();
    let a_indices: Vec<u32> = assigned.col_indices().iter().map(|&i| i as u32).collect();

    // input X CSR — preserved on the output AnnData so the user gets the
    // original counts plus the assignment layer in one object.
    let x_csr = input.counts.csr();
    let x_data: Vec<u32> = x_csr.values().to_vec();
    let x_indices: Vec<u32> = x_csr.col_indices().iter().map(|&i| i as u32).collect();
    let x_indptr: Vec<u32> = x_csr.row_offsets().iter().map(|&i| i as u32).collect();

    let cell_barcodes: Vec<String> = (0..input.covariates.cell_barcodes.len())
        .map(|i| input.covariates.cell_barcodes.value(i).to_string())
        .collect();
    let guide_ids: Vec<String> = (0..input.guide_metadata.guide_ids.len())
        .map(|i| input.guide_metadata.guide_ids.value(i).to_string())
        .collect();

    let batch_py = PyRecordBatch::new(result.batch).to_pyarrow(py)?;

    Ok((
        batch_py,
        a_indptr.into_pyarray_bound(py),
        a_indices.into_pyarray_bound(py),
        x_data.into_pyarray_bound(py),
        x_indices.into_pyarray_bound(py),
        x_indptr.into_pyarray_bound(py),
        cell_barcodes,
        guide_ids,
        model_name,
        model_params_json,
    ))
}

/// The compiled module lives at `kaichi._native`; the pure-Python wrapper at
/// `python/kaichi/__init__.py` imports from it.
#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_assign_from_h5ad_inmem, m)?)?;
    Ok(())
}

//! Python bindings for kaichi.
//!
//! Exposes three private functions consumed by the Python wrapper at
//! `kaichi-py/python/kaichi/__init__.py`:
//!
//! - `_assign_from_h5ad_inmem` — one-shot assign (all models); returns the
//!   existing 10-tuple for backwards compatibility with the v0.1 Python layer.
//! - `_score` — two-stage models only; returns a `PyScoreResult` (zero-copy).
//! - `_decide` — threshold a `PyScoreResult`; returns an Arrow RecordBatch.

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
    AssignmentModel, TwoStage,
};
use kaichi_core::score::ScoreMatrix;

use arrow::array::Array as _;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;
use std::path::Path;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// PyScoreResult
// ---------------------------------------------------------------------------

/// Python-facing wrapper around a Rust-owned `ScoreMatrix`.
///
/// All Arrow arrays inside are Arc-backed; returning them to Python via
/// `pyo3_arrow` bumps the refcount — no buffer copies.
#[pyclass(frozen)]
pub struct PyScoreResult {
    inner: ScoreMatrix,
}

#[pymethods]
impl PyScoreResult {
    /// Float32 posterior scores, one per nonzero entry. Length = nnz.
    #[getter]
    fn scores_data(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.data.clone())).to_pyarrow(py)
    }

    /// Preserved UMI counts, parallel to `scores_data`. Length = nnz.
    #[getter]
    fn umi_counts(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.umi_counts.clone())).to_pyarrow(py)
    }

    /// CSR column indices (guide index for each nonzero entry). Length = nnz.
    #[getter]
    fn indices(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.indices.clone())).to_pyarrow(py)
    }

    /// CSR row pointers. Length = n_cells + 1.
    #[getter]
    fn indptr(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.indptr.clone())).to_pyarrow(py)
    }

    #[getter]
    fn cell_barcodes(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.cell_barcodes.clone())).to_pyarrow(py)
    }

    #[getter]
    fn guide_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3_arrow::PyArray;
        PyArray::from_array_ref(Arc::new(self.inner.guide_ids.clone())).to_pyarrow(py)
    }

    /// (n_cells, n_guides)
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.n_cells(), self.inner.n_guides())
    }

    #[getter]
    fn model(&self) -> &'static str {
        self.inner.model.name()
    }

    #[getter]
    fn model_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let json_str = serde_json::to_string(&self.inner.model_params)
            .map_err(|e| PyValueError::new_err(format!("model_params: {e}")))?;
        let json_mod = py.import_bound("json")?;
        json_mod.call_method1("loads", (json_str,)).map(|o| o.unbind())
    }
}

// ---------------------------------------------------------------------------
// _score
// ---------------------------------------------------------------------------

/// Score a guide-count h5ad with a two-stage model. Returns a `PyScoreResult`.
/// Raises `ValueError` for single-stage models (umi, ratio, max).
#[pyfunction]
#[pyo3(signature = (h5ad_path, model, *, n_jobs=None))]
fn _score(
    _py: Python<'_>,
    h5ad_path: &str,
    model: &str,
    n_jobs: Option<usize>,
) -> PyResult<PyScoreResult> {
    let input = read_h5ad(Path::new(h5ad_path))
        .map_err(|e| PyIOError::new_err(format!("read_h5ad({h5ad_path}): {e:#}")))?;

    let pool = make_pool(n_jobs)?;

    let score_matrix = pool.install(|| score_model(model, &input))?;
    Ok(PyScoreResult { inner: score_matrix })
}

// ---------------------------------------------------------------------------
// _decide
// ---------------------------------------------------------------------------

/// Apply a confidence threshold to a `PyScoreResult`. Returns an Arrow RecordBatch.
#[pyfunction]
#[pyo3(signature = (scores, min_confidence))]
fn _decide(
    py: Python<'_>,
    scores: PyRef<'_, PyScoreResult>,
    min_confidence: f32,
) -> PyResult<PyObject> {
    // Cast to usize to cross the Ungil bound; safe because `scores` (the PyRef)
    // remains on the caller stack and keeps the PyScoreResult alive for the full duration.
    let inner_addr = (&scores.inner as *const ScoreMatrix) as usize;
    let result = py
        .allow_threads(|| {
            let inner = unsafe { &*(inner_addr as *const ScoreMatrix) };
            kaichi_core::models::em_count_mixture::decide_threshold(inner, min_confidence)
        })
        .map_err(|e| PyValueError::new_err(format!("decide: {e:#}")))?;
    PyRecordBatch::new(result.batch).to_pyarrow(py)
}

// ---------------------------------------------------------------------------
// _assign_from_h5ad_inmem  (v0.1 API, preserved for Python layer compatibility)
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (h5ad_path, model, *, min_confidence=None, quantile=None, n_jobs=None))]
fn _assign_from_h5ad_inmem<'py>(
    py: Python<'py>,
    h5ad_path: &str,
    model: &str,
    min_confidence: Option<f32>,
    quantile: Option<f32>,
    n_jobs: Option<usize>,
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

    let pool = make_pool(n_jobs)?;
    let (result, model_name, model_params_json) =
        pool.install(|| run_model(model, &input, min_confidence, quantile))?;

    extract(py, input, result, model_name, model_params_json)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_pool(n_jobs: Option<usize>) -> PyResult<rayon::ThreadPool> {
    let n_threads = match n_jobs {
        Some(n) if n > 0 => n,
        _ => {
            let total = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            (total / 2).max(1)
        }
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("rayon pool init failed: {e}")))
}

/// Score a two-stage model. Raises ValueError for single-stage models.
fn score_model(name: &str, input: &LoadedInput) -> PyResult<ScoreMatrix> {
    macro_rules! score {
        ($model_expr:expr) => {{
            let m = $model_expr;
            m.score(input).map_err(|e| PyValueError::new_err(format!("score: {e:#}")))
        }};
    }
    match name {
        "poisson_gauss" => score!(PoissonGaussModel::default()),
        "poisson" => score!(PoissonModel::default()),
        "neg_binomial" => score!(NegBinomialModel::default()),
        "binomial" => score!(BinomialModel::default()),
        "umi" | "ratio" | "max" => Err(PyValueError::new_err(format!(
            "model {name:?} is single-stage and does not support score(). \
             Use kaichi.assign() instead."
        ))),
        other => Err(PyValueError::new_err(format!(
            "unknown model {other:?}: expected one of poisson_gauss/poisson/neg_binomial/\
             binomial/beta2/beta3/quantiles for score(), or any model for assign()"
        ))),
    }
}

/// Dispatch on model name for one-shot assign.
fn run_model(
    name: &str,
    input: &LoadedInput,
    min_confidence: Option<f32>,
    quantile: Option<f32>,
) -> PyResult<(AssignmentResult, &'static str, String)> {
    macro_rules! run {
        ($model_expr:expr) => {{
            let m = $model_expr;
            let res = m.assign(input).map_err(|e| PyValueError::new_err(format!("model.assign failed: {e:#}")))?;
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
    let assigned = &result.assigned_x;
    let a_indptr: Vec<u32> = assigned.row_offsets().iter().map(|&i| i as u32).collect();
    let a_indices: Vec<u32> = assigned.col_indices().iter().map(|&i| i as u32).collect();

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

const VERSION: &str = git_version::git_version!(fallback = env!("CARGO_PKG_VERSION"));

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(_assign_from_h5ad_inmem, m)?)?;
    m.add_function(wrap_pyfunction!(_score, m)?)?;
    m.add_function(wrap_pyfunction!(_decide, m)?)?;
    m.add_class::<PyScoreResult>()?;
    Ok(())
}

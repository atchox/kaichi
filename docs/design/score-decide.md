# kaichi — Score / Decide Split (v0.2)

For the EM mixture models (`poisson_gauss`, `poisson`, `neg_binomial`,
`binomial`, `beta2`, `beta3`) and `quantiles`, the current `assign()` API
conflates two operations:

1. **Score**: produce a per-(cell, guide) score — signal-component posterior
   for the EM models, rank percentile for `quantiles`. **Expensive.**
2. **Decide**: apply a threshold to the score to produce assignments. **Cheap.**

Right now they're fused. To sweep `min_confidence` on a 100k-cell dataset, you
re-fit the EM mixture every time. This document defines the split.

The split does **not** apply to the threshold-only models (`umi`, `ratio`,
`max`) — for them the "score" is either the raw count or a trivially-computed
ratio and caching it buys nothing. Those keep the single-stage API. The exact
matrix of which models split, and the public-API consequences, are in the
*Which models support the split* table below.

---

## Goals

- **One expensive Rust pass** produces a score; many cheap decisions can follow.
- **Zero-copy in Python** when chaining `score()` → `decide()`. The user must not
  pay a serialization round-trip across the FFI boundary for the intermediate.
- **Minimal regression for the one-shot case**. `assign(path, model, min_confidence)`
  must not cross the Python FFI more than once, and must not materialise any
  extra Python object compared to today. It does materialise the intermediate
  `ScoreMatrix` on the Rust stack (~80 MB for a 100k × 1000 dataset with
  ~10M nonzeros), which today's fully-fused per-cell loop avoids — this is
  the cost of the split and is judged acceptable; see *About the one-shot
  path* below.
- **CLI parity with single files**. `kaichi score` writes an `.h5ad`,
  `kaichi decide` reads one. The two CLI subcommands compose via disk; there is
  no in-memory shortcut at the CLI layer (and there doesn't need to be — CLI
  invocations are independent processes).
- **Result identity**: `score → decide` must produce the same assignment table
  as the fused `assign`, byte-equal for deterministic models and within
  `1e-6` (per [validation.md](validation.md)) for EM models with the same seed.

---

## CLI surface

```bash
# Stage 1: scoring (expensive) — two-stage models only
kaichi score \
    --counts gRNA_counts.h5ad \
    --model poisson_gauss \
    --output scores.h5ad

# Stage 2: decision (cheap; rerun with different thresholds freely)
kaichi decide \
    --scores scores.h5ad \
    --min-confidence 0.8 \
    --output assignments.h5ad

# One-shot — works for all models
kaichi assign \
    --counts gRNA_counts.h5ad \
    --model poisson_gauss \
    --min-confidence 0.8 \
    --output assignments.h5ad
```

`kaichi assign` is the universal entry point and works with every model. For
two-stage models it is exactly `score()` followed by `decide()` in the same
process with no h5ad round-trip (same as the Python `assign()`). For
single-stage models (`umi`, `ratio`, `max`) it goes through the direct path.

`kaichi score --model umi` (or `ratio`, `max`) is rejected with an error: those
models don't have a meaningful score artefact, so writing one would be misleading.
The error message points the user to `kaichi assign`.

The `--output` format detection (`.h5ad` vs `.csv`) only applies to `decide`
and `assign`; `score` always writes `.h5ad`.

### `scores.h5ad` on-disk layout

```
scores.h5ad
├── X              float32 sparse CSR (cells × guides)   — the score matrix
├── obs            DataFrame indexed by cell_barcode
├── var            DataFrame indexed by guide_id
└── uns/kaichi
    ├── stage      "scores"
    ├── model      "poisson_gauss"
    ├── model_params  {max_em_iters: 200, n_restarts: 4, tol: 1e-4, ...}
    │                                          (score-time params only — no decision
    │                                          thresholds, which are decide-time)
    └── version    "v0.2.0"                  (or `git describe` between tags)
```

`uns/kaichi/model` is mandatory — `decide` reads it to choose the right
decision rule (the score *value* alone doesn't tell you whether to threshold
at 0.8 or 5 UMIs). `model_params` is informational only.

`X` is stored as CSR sparse with sparsity pattern matching the original counts
matrix — only `(cell, guide)` pairs with `count > 0` carry a score. See
*Why sparse-by-pattern* below for the math justification.

### Which models support the split

The split only makes sense when **score generation is expensive** (so caching
it pays off) and **the threshold is genuinely tunable** (so there's a reason to
re-decide). That's not every model.

| Model | Two-stage? | Score | Decision rule |
|---|---|---|---|
| `poisson_gauss` | ✓ | signal posterior ∈ [0, 1] | `score ≥ min_confidence` |
| `poisson` | ✓ | signal posterior ∈ [0, 1] | `score ≥ min_confidence` |
| `neg_binomial` | ✓ | signal posterior ∈ [0, 1] | `score ≥ min_confidence` |
| `binomial` | ✓ | signal posterior ∈ [0, 1] | `score ≥ min_confidence` |
| `beta2` | ✓ | high-component posterior | `score ≥ min_confidence` |
| `beta3` | ✓ | high-component posterior | `score ≥ min_confidence` |
| `quantiles` | ✓ | rank percentile per guide | top Q% per guide |
| `umi` | ✗ | — | (direct: `count ≥ n_umis`) |
| `ratio` | ✗ | — | (direct: top/total ≥ fraction) |
| `max` | ✗ | — | (direct: per-cell argmax) |

The ✓ models do real work in `score` — EM fitting, or full-population ranking
for `quantiles` — and that work is shareable across many `decide` calls.

The ✗ models are **decision-only**. Their "score" would be either the raw
count (`umi`, `max`) or a trivially-computed ratio. Pre-computing it buys
nothing, and exposing `score()` on them would just be ceremony. So:

- `kaichi.score(path, model="umi")` → raises `ValueError` with a message
  explaining that `umi` is a single-stage model; use `kaichi.assign(...)`.
- Same for `kaichi score --model umi` at the CLI.
- `assign()` works for **all** models — it dispatches to the two-stage pipeline
  for ✓ models and the direct pipeline for ✗ models, transparently.

The score matrix has uniform `float32` dtype across all two-stage models. The
EM-mixture scores are already in [0, 1]; `quantiles` percentiles are in [0, 1]
too. No model needs more than float32 of precision.

### Why sparse-by-pattern

For the four count-based EM models (`poisson`, `neg_binomial`, `binomial`,
`poisson_gauss`), the score matrix is stored with the **same sparsity pattern
as the input counts**: only `(cell, guide)` pairs with `count > 0` carry a
posterior. This isn't an arbitrary optimization; it's load-bearing
mathematically and biologically.

**Numerically:** for pure-Poisson signal (`poisson`, `neg_binomial`,
`binomial`) the signal mean μ_sig is many times the background mean μ_bg, so:

```
P(signal | y=0) = π · Pois(0; μ_sig) / [π · Pois(0; μ_sig) + (1−π) · Pois(0; μ_bg)]
                = π · exp(−μ_sig) / [π · exp(−μ_sig) + (1−π) · exp(−μ_bg)]
                ≈ 10⁻⁹ for typical fits (μ_sig=20, μ_bg=0.1, π=0.05)
```

Storing this posterior takes 4 bytes per cell-guide pair to record a value
that no threshold check could ever pass.

`poisson_gauss` has a Gaussian signal whose tail at y=0 is fatter (~0.005
instead of ~10⁻⁹ at the same parameters), pushing `P(signal | 0)` up to
~10⁻⁴. Still well below any threshold a user would set, and the existing
implementation has an explicit safeguard ([poisson_gauss.rs:152-153](../../kaichi-core/src/models/poisson_gauss.rs#L152-L153))
that derives a per-guide integer threshold ≥ 1, structurally excluding y=0
from assignment regardless.

**Biologically:** assigning a guide to a cell with zero UMIs for it is
nonsense — the cell measurably did not receive that guide. The existing
per-cell loops bake this invariant in by walking only nonzero positions
([poisson.rs:93](../../kaichi-core/src/models/poisson.rs#L93)). The sparsity
pattern of the score matrix should match that loop's domain.

**Consequence:** for 100k cells × 1000 guides with ~10M nonzero counts,
sparse-by-pattern is **~80 MB instead of 400 MB** in memory (see the cost
table in *Memory layout* for the exact breakdown) and similarly smaller on
disk after HDF5 compression. Densification on read would put structural
zeros back into a representation the algorithm refuses to use anyway.

The Beta models (`beta2`, `beta3`) and `quantiles` have different per-pair
semantics — Beta scores per-cell-max-proportion (one value per cell, not per
pair); `quantiles` ranks each guide column over the full cell population.
Their storage layouts are model-specific and documented per-model.

---

## Python surface

```python
import kaichi

# Stage 1
scores = kaichi.score("gRNA_counts.h5ad", model="poisson_gauss", n_jobs=8)
# type: kaichi.ScoreResult

# Stage 2 (cheap, sweep at will)
a1 = kaichi.decide(scores, min_confidence=0.7)
a2 = kaichi.decide(scores, min_confidence=0.9)
# type of decide() output: pyarrow.Table (same shape as today's assign())

# One-shot — no intermediate Python object
a = kaichi.assign(
    "gRNA_counts.h5ad",
    model="poisson_gauss",
    min_confidence=0.8,
)
```

### `ScoreResult` — the in-memory format

`ScoreResult` is a **PyO3 `#[pyclass]` wrapping a Rust-owned `ScoreMatrix`
that is itself backed by Arrow buffers**. Per the binding-interop principle
(Arrow at the boundary), nothing crosses the FFI as ad-hoc tuples or
language-native types — every cross-language artefact is an Arrow array.

Why a `ScoreResult` wrapper rather than returning a bare `pyarrow.Table`:

1. Passing the same `ScoreResult` to `decide()` is a `PyRef` borrow: Rust
   reaches into the same `ScoreMatrix` allocated by `score()`. No
   re-conversion, no buffer copy.
2. Inspection stays ergonomic: `scores.to_anndata()` materialises an AnnData
   (with the score matrix as zero-copy `X`) for users who want scanpy-friendly
   objects, but `decide()` never pays that cost.
3. The same Rust struct underlies the R binding's analogous wrapper class —
   one source of truth for the cross-language contract.

```python
class ScoreResult:
    # The sparse score matrix as three Arrow arrays (the standard CSR convention
    # from binding-interop.md). All zero-copy via the PyCapsule protocol.
    @property
    def values_csr(self) -> pyarrow.RecordBatch: ...
        # schema: data    Float32Array  (length nnz)
        #         indices UInt32Array   (length nnz, guide_idx per nonzero)
        #         indptr  UInt32Array   (length n_cells + 1)

    @property
    def cell_barcodes(self) -> pyarrow.StringArray: ...
    @property
    def guide_ids(self) -> pyarrow.StringArray: ...

    # Shape (cheap, derived from the CSR arrays)
    @property
    def shape(self) -> tuple[int, int]: ...                # (n_cells, n_guides)

    # Plain Python for small metadata
    @property
    def model(self) -> str: ...                            # e.g. "poisson_gauss"
    @property
    def model_params(self) -> dict: ...                    # parsed from JSON, cached

    # Conveniences
    def to_scipy_csr(self) -> scipy.sparse.csr_matrix: ... # zero-copy: wraps the same buffers
    def to_anndata(self) -> anndata.AnnData: ...           # X is the scipy CSR view
    def write_h5ad(self, path: str | Path) -> None: ...

    @classmethod
    def read_h5ad(cls, path: str | Path) -> "ScoreResult": ...
```

The score matrix is **sparse CSR** with the same sparsity pattern as the input
counts (one entry per `(cell, guide)` pair with `count > 0`). The three Arrow
arrays follow the convention already used for counts in
[binding-interop.md](binding-interop.md#sparse-matrix-handoff-three-arrays-one-recordbatch).

`to_scipy_csr()` wraps the same three buffers as a `scipy.sparse.csr_matrix`
without copying — scipy's CSR constructor accepts numpy views of `data`,
`indices`, `indptr`. AnnData then accepts that scipy object directly as `X`.

### Why Arrow, not numpy/scipy as the primary

Numpy/scipy is the convenient choice in Python but it's a Python-only
currency. The boundary contract in [binding-interop.md](binding-interop.md)
is that Rust↔language data crosses as Arrow. The R binding cannot use the
numpy crate; if `ScoreResult` were numpy/scipy-first, the R binding would
need a parallel design. Arrow CSR gives:

- Zero-copy Rust → Python via pyo3-arrow's PyCapsule protocol
- Zero-copy Rust → R via extendr-arrow / nanoarrow
- A `to_scipy_csr()` convenience that's still zero-copy for Python users
- Consistent treatment with the counts input (already CSR-via-three-arrays)
  and the assignment output (already a `pyarrow.Table`)

### Why not just return an `AnnData`?

The AnnData approach would force `decide()` to either:

- (a) Accept an AnnData and reach into `.X` and `.uns["kaichi"]["model"]`. Now
      `decide()`'s contract is "AnnData with a specific structure", which is
      easy to break and hard to type-check.
- (b) Take separate args (`scores`, `model`, `cell_barcodes`, …), forcing the
      user to unpack on every call.

`ScoreResult` is the same idea as `pyarrow.Table` or `polars.DataFrame`:
a typed handle backed by zero-copy native (Arrow) buffers, with `to_pandas()` /
`to_anndata()` for the cases where the user actually wants the heavier object.

### Cross-language note

`ScoreResult` is a Python-only type. The Rust core operates on
`kaichi_core::ScoreMatrix`. The R binding will expose its own thin wrapper
class around the same `ScoreMatrix` (extendr `#[extendr]` types, analogous to
PyO3 `#[pyclass]`), with `values` returning an arrow R object. Because the
underlying buffers are Arrow on both sides, there is no language-specific
conversion logic — the Rust struct is the shared contract; the wrappers are
thin façades.

---

## Memory layout

### Rust core

```rust
pub struct ScoreMatrix {
    /// CSR sparse storage. Sparsity pattern matches the input counts.
    /// Three Arrow primitive arrays, all backed by Arc'd buffers.
    pub data:    arrow::array::Float32Array,   // length nnz
    pub indices: arrow::array::UInt32Array,    // length nnz (guide_idx per nonzero)
    pub indptr:  arrow::array::UInt32Array,    // length n_cells + 1

    pub cell_barcodes: arrow::array::StringArray,   // length n_cells
    pub guide_ids:     arrow::array::StringArray,   // length n_guides

    pub model: ModelKind,                // enum, cheap to clone
    pub model_params: serde_json::Value, // provenance only; `decide` doesn't read this
}

impl ScoreMatrix {
    pub fn n_cells(&self)  -> usize { self.indptr.len() - 1 }
    pub fn n_guides(&self) -> usize { self.guide_ids.len() }
    pub fn nnz(&self)      -> usize { self.data.len() }

    /// Iterate (guide_idx, score) pairs for one cell. Zero allocation.
    pub fn row(&self, cell: usize) -> impl Iterator<Item = (u32, f32)> + '_ {
        let lo = self.indptr.value(cell) as usize;
        let hi = self.indptr.value(cell + 1) as usize;
        (lo..hi).map(|k| (self.indices.value(k), self.data.value(k)))
    }
}
```

The per-cell decision loop walks `row(cell)` exactly the way today's per-cell
loop walks `counts.csr().get_row(cell)`. The two loops have the same shape
because the score sparsity pattern equals the counts sparsity pattern.

### Python binding (PyO3)

```rust
#[pyclass(frozen)]
pub struct PyScoreResult {
    inner: ScoreMatrix,  // moved in at construction; never replaced
}

#[pymethods]
impl PyScoreResult {
    /// Arrow-native handoff. The three CSR arrays go out as a RecordBatch
    /// via pyo3-arrow's PyCapsule protocol — zero copy. The `.clone()`
    /// calls are Arc refcount bumps, not buffer copies.
    #[getter]
    fn values_csr(&self, py: Python) -> PyResult<PyObject> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("data",    DataType::Float32, false),
            Field::new("indices", DataType::UInt32,  false),
            Field::new("indptr",  DataType::UInt32,  false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![
            Arc::new(self.inner.data.clone()),
            Arc::new(self.inner.indices.clone()),
            Arc::new(self.inner.indptr.clone()),
        ])?;
        pyo3_arrow::PyRecordBatch::from(batch).to_arro3(py)
    }

    #[getter]
    fn cell_barcodes(&self, py: Python) -> PyResult<PyObject> {
        pyo3_arrow::PyArray::from_array(self.inner.cell_barcodes.clone()).to_arro3(py)
    }

    #[getter]
    fn guide_ids(&self, py: Python) -> PyResult<PyObject> {
        pyo3_arrow::PyArray::from_array(self.inner.guide_ids.clone()).to_arro3(py)
    }

    #[getter]
    fn shape(&self) -> (usize, usize) { (self.inner.n_cells(), self.inner.n_guides()) }

    fn to_scipy_csr(&self, py: Python) -> PyResult<PyObject> { ... }
    fn to_anndata(&self, py: Python) -> PyResult<PyObject> { ... }
    fn write_h5ad(&self, path: &str) -> PyResult<()> { ... }
}

// Score: takes a path, returns the wrapper.
#[pyfunction]
fn _score(py: Python, path: &str, model: &str, ...) -> PyResult<PyScoreResult> {
    let input = read_h5ad(path)?;
    let score_matrix = py.allow_threads(|| fit_and_score(input, model))?;
    Ok(PyScoreResult { inner: score_matrix })
}

// Decide: borrows the wrapper through PyRef, returns a record batch.
#[pyfunction]
fn _decide(py: Python, scores: PyRef<PyScoreResult>, min_confidence: f32)
    -> PyResult<PyRecordBatch>
{
    let inner_ptr: *const ScoreMatrix = &scores.inner;
    let assignments = py.allow_threads(|| {
        // Safe: `scores` (the PyRef) is alive on the caller stack for the
        // duration of this function, so the inner ScoreMatrix is alive too.
        let inner = unsafe { &*inner_ptr };
        score_to_assignments(inner, min_confidence)
    })?;
    Ok(record_batch_from_assignments(assignments))
}

// Assign: same as score() followed by decide(), but the ScoreMatrix never
// gets wrapped as a PyClass — it lives and dies on the Rust stack.
#[pyfunction]
fn _assign(py: Python, path: &str, model: &str, min_confidence: f32, ...)
    -> PyResult<PyRecordBatch>
{
    let input = read_h5ad(path)?;
    let assignments = py.allow_threads(|| {
        let scores = fit_and_score(input, model)?;
        score_to_assignments(&scores, min_confidence)
        // `scores` drops here without ever being seen by Python.
    })?;
    Ok(record_batch_from_assignments(assignments))
}
```

The raw-pointer dance in `_decide` is the standard PyO3 way to thread a
GIL-bound borrow through `allow_threads`: extract the pointer ahead of the
release, deref inside. `scores` (the `PyRef`) is on the caller's Rust stack
for the entire duration, so the pointer is valid.

**About the one-shot path**: `_assign` does still allocate the full `ScoreMatrix`
on the stack before decide consumes it. Today's fused `assign()` doesn't —
it computes posteriors inline in the per-cell loop and never materialises
the full matrix. With sparse-by-pattern the intermediate is ~60 MB rather
than ~400 MB, freed before return, which is a small enough regression to
accept for the simpler architecture. If the materialisation cost ever
matters, the trait can later expose an `assign_streaming` path that fuses
score and decide per row.

### Lifetime / drop sequence

```
scores = kaichi.score(...)        # ref 1 (Python → PyScoreResult)
a1 = kaichi.decide(scores, 0.7)   # PyRef borrow, released at return
a2 = kaichi.decide(scores, 0.9)   # PyRef borrow, released at return
view = scores.values_csr          # pyarrow RecordBatch holds capsules → PyScoreResult (ref 2)
del scores                        # ref 1 dropped; view still holds ref 2
del view                          # ref 2 dropped → PyScoreResult.__dealloc__
                                  # → ScoreMatrix dropped → Arrow buffers freed
```

PyO3 refcounting on the Python side, standard `Drop` on the Rust side. The
PyCapsule that pyo3-arrow attaches to each exported Arrow array holds a
strong reference to `PyScoreResult`, so the buffers survive as long as any
view of them does.

### Cost summary for chained Python use

| Call | Rust → Python boundary crossings | Buffer copies |
|---|---|---|
| `assign(path, model, t)` | 1 (record batch out) | 0 |
| `score(path, model)` + `decide(s, t)` | 2 (`PyScoreResult` out, then record batch out) | 0 |
| `score(path, model)` + 10× `decide(s, t_i)` | 11 | 0 |
| `score(path, model)` + `scores.to_scipy_csr()` + 10× `decide(s, t_i)` | 12 | 0 |

Every additional `decide` is just a `PyRef` borrow and an in-Rust per-row
walk over the same sparse Arrow buffers.

For a 100k cell × 1000 guide dataset with ~10M nonzeros, the resident memory
footprint of the cached `ScoreResult` is:

- `data`    (f32): 40 MB
- `indices` (u32): 40 MB
- `indptr`  (u32): 400 KB
- barcodes + guide_ids: ~2 MB

≈ **80 MB** total, vs. **400 MB** for a dense `n_cells × n_guides` matrix.

---

## Trait split (kaichi-core)

The current `AssignmentModel::assign(&input) -> AssignmentResult` is replaced by
two traits and a unified entry point:

```rust
/// Single-stage interface. Implemented by every model.
pub trait AssignmentModel {
    fn assign(&self, input: &LoadedInput) -> Result<AssignmentResult>;
}

/// Two-stage interface. Implemented only by models where score caching pays.
pub trait TwoStage: AssignmentModel {
    type Threshold;

    fn score(&self, input: &LoadedInput) -> Result<ScoreMatrix>;
    fn decide(&self, scores: &ScoreMatrix, threshold: Self::Threshold)
        -> Result<AssignmentResult>;
}
```

Models implementing `TwoStage` get their `AssignmentModel::assign` provided
automatically as `self.decide(&self.score(input)?, threshold)`. Decision-only
models implement only `AssignmentModel::assign` directly.

```rust
// Two-stage:  poisson_gauss, poisson, neg_binomial, binomial, beta2, beta3, quantiles
// Direct:     umi, ratio, max
```

The dispatch table (currently `model_from_name`) returns a `Box<dyn AssignmentModel>`.
A separate `two_stage_from_name` returns `Box<dyn TwoStage>` for callers that
need the split (the `score` CLI subcommand and `kaichi.score()`); it errors for
`umi`, `ratio`, `max`.

Per-model notes:

- **EM models** (`poisson_gauss`, `poisson`, `neg_binomial`, `binomial`, `beta2`,
  `beta3`) keep their fitting state — init strategy, restart count, max iters —
  in the `TwoStage::score` impl. `decide` is uniform across them:
  `score ≥ min_confidence` with the multi-infection logic in
  [assignment-models.md](assignment-models.md).
- **`quantiles`** scores by ranking each guide column over the full cell
  population. `Threshold = f32` (the quantile).
- **`umi`, `ratio`, `max`** are direct-only. Their `assign()` reads counts and
  produces the assignment table in a single pass.

---

## Refactor opportunity: collapse duplication in EM mixture models

Separating score from decide forces the per-cell scoring loop and per-cell
decision loop into shared helpers. Once those are extracted, the existing
per-model files lose most of their bulk — today each Poisson-family EM model
duplicates ~150 lines of identical orchestration. Doing the split without
also doing the trait extraction means refactoring the same hot loop twice.

### What's duplicated today

Across [poisson.rs](../../kaichi-core/src/models/poisson.rs),
[neg_binomial.rs](../../kaichi-core/src/models/neg_binomial.rs),
[binomial.rs](../../kaichi-core/src/models/binomial.rs), and
[poisson_gauss.rs](../../kaichi-core/src/models/poisson_gauss.rs):

| Block | Status |
|---|---|
| Extract csc, log_depths, batch | identical |
| Parallel per-guide fit loop | identical, only the `fit_guide` callee differs |
| Multistart EM driver (`fit_mixture`) | identical structure |
| Per-cell scoring loop (csr walk, posterior, threshold, best-guide, multi-infection) | identical, only `posterior_signal` callee differs |
| Output builder calls | identical |

The actually-different code per model is: `initialize_params`, `m_step`,
`posterior_signal`, and the gating thresholds. That's the math; everything
else is plumbing.

### The trait that captures only the differences

```rust
pub trait EmCountMixture {
    type FitParams: Clone + Send;

    fn gating(&self) -> Gating;   // min_nonzero, min_max_count
    fn em_config(&self) -> EmConfig;  // max_iters, n_restarts, tol

    fn init(&self, data: &GuideData, n_batches: usize) -> Self::FitParams;
    /// One full E→M cycle. Returns (new_params, log_likelihood).
    fn step(&self, params: Self::FitParams, data: &GuideData, n_batches: usize)
        -> (Self::FitParams, f64);
    fn posterior(&self, count: u32, log_d: f64, params: &Self::FitParams, batch: u16) -> f32;
}
```

The Beta models (`beta2`, `beta3`) don't take `(count, log_d, batch)` — they
score per-cell proportions. Either factor out a sibling trait
`EmProportionMixture` or generalise the input type as an associated type. The
former is cleaner; the genericity isn't worth the type gymnastics.

### One generic `score` and one generic `decide`

```rust
/// Used by every Poisson-family EM model.
pub fn score_em_count_mixture<M: EmCountMixture>(
    model: &M,
    input: &LoadedInput,
) -> Result<ScoreMatrix> {
    // log_depths + batch extraction (was duplicated 4×)
    // parallel per-guide fit via run_em_multistart + run_em + model.step (was duplicated 4×)
    // per-cell posterior fill into CSR data/indices arrays (was duplicated 4×)
}

/// Used by every posterior-based model (Poisson family + Beta family + quantiles).
pub fn decide_threshold(
    scores: &ScoreMatrix,
    min_conf: f32,
) -> Result<AssignmentResult> {
    // per-cell best-guide tracking + multi-infection logic + output builder
    // (was duplicated 6×)
}
```

Per-model files shrink to: struct + `Default` + config + the three math
functions + `impl EmCountMixture`. Approximate footprint:

| File | Today | After |
|---|---|---|
| `poisson.rs` | 558 | ~200 |
| `neg_binomial.rs` | 634 | ~250 (Newton step for θ is real) |
| `binomial.rs` | 489 | ~180 |
| `poisson_gauss.rs` | 580 | ~250 |
| `beta.rs` (`beta2` + `beta3`) | 869 | ~450 |

Net: ~2.7k lines → ~1.0k lines of EM mixture code, plus the new generic
helpers (~250 lines). One canonical per-cell scoring loop, one canonical
decision loop. Bug fixes in either propagate to all models automatically.
Adding a new EM family becomes "implement the trait, add to the dispatch
table" — no copy-paste.

### Sequencing

Do the trait extraction **as part of** the score/decide split, not in a
separate pass. The score/decide split needs the per-cell loops extracted
anyway; doing it without unifying the per-model files leaves the duplication
in place under a thin trait veneer.

---

## What this changes in existing docs

These design docs need updates after implementation lands:

- [io-spec.md](io-spec.md) — add the `scores.h5ad` layout and the two new
  subcommands.
- [binding-interop.md](binding-interop.md) — replace the v0.1 "single
  `_assign_from_h5ad_inmem` returning a tuple" with the new triplet
  (`_score`, `_decide`, `_assign`) and the `PyScoreResult` wrapper.
- [cli.md](cli.md) — add `kaichi score` and `kaichi decide`.
- [data-model.md](data-model.md) — document the score matrix as a first-class
  on-disk artifact.

---

## Open questions

1. **Beta / `quantiles` score layout.** The count-based EM models share one
   CSR-by-pattern layout. Beta models produce one scalar per cell (per-cell
   max proportion) and `quantiles` produces one dense column per guide. These
   need explicit per-model layouts in `ScoreMatrix` — likely an enum variant
   (`Csr`, `DensePerCell`, `DensePerGuide`) chosen by the model. Defer the
   final enum design until we implement those models in the new framework.
2. **Multiple-decision output.** Should `decide()` accept an iterable of
   thresholds and return a list of tables, so a sweep is one FFI call? Only
   worth it if FFI overhead dominates, which it won't until thresholds-per-call
   exceeds ~hundreds.
3. **Round-trip identity** across `score(disk) → decide` vs `score(memory) → decide`.
   The h5ad write/read should be lossless for the score matrix (`float32` in,
   `float32` out). Add an equivalence test to [validation.md](validation.md).
4. **`assign_streaming` for one-shot.** If the ~80 MB intermediate
   `ScoreMatrix` in the one-shot path ever becomes a problem (very large
   datasets, memory-constrained nodes), add a `TwoStage::assign_streaming`
   default impl that fuses score and decide per-row, never materialising the
   full score matrix. Not needed for v0.2.
5. **pyo3-arrow version.** ✓ Resolved. `pyo3-arrow` 0.5 on `arrow` 53 supports
   exporting a `RecordBatch` of three primitive arrays (`Float32Array`,
   `UInt32Array`, `UInt32Array`) to pyarrow via the Arrow C Data Interface
   (PyCapsule protocol) — zero copy. Verified against pyo3-arrow source and
   the existing `PyRecordBatch::to_pyarrow` use at
   [kaichi-py/src/lib.rs:200](../../kaichi-py/src/lib.rs#L200).

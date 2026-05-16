# kaichi — Binding Interop

How data crosses the Rust ↔ Python and Rust ↔ R boundaries. The contract here is
what makes "one Rust core, two bindings, identical results" actually work.

> **Status note**: the architecture described from "The contract" onward is the
> target end-state. The actual v0.1 Python binding we're shipping first is
> narrower and pragmatic; it's documented in the **Python v0.1** section below.
> The v0.1 design will grow toward the full Arrow-native contract over time
> rather than landing all at once.

---

## Python v0.1 — what we're actually building

### Surface

A single user-facing function:

```python
import kaichi
adata = kaichi.assign("input.h5ad", model="poisson", **model_kwargs)
# adata is an in-memory anndata.AnnData with:
#   .X                       — preserved raw UMI counts (cells × guides)
#   .layers["assigned"]      — binary CSR, 1 where (cell, guide) was assigned
#   .obs["guide_id"]         — assigned guide ID (categorical, missing if unassigned)
#   .obs["assignment_confidence"]
#   .obs["umi_count"]
#   .obs["is_unassigned"]
#   .obs["is_multi_infected"]
#   .obs["n_guides_detected"]
#   .uns["kaichi"]           — {"model", "model_params", "version"}
```

The user then does whatever they want with the AnnData: attach to existing
state, write h5ad/h5mu, hand to scanpy, etc.

### Why path-in / AnnData-out (not Arrow-in / Arrow-out yet)

The aspirational architecture below has the Rust core consume Arrow `RecordBatch`
inputs from the binding. That requires refactoring `kaichi_core::io::read::read_h5ad`
into "read into Arrow batches" + a separate "Arrow batches → fitting" path. Today
the core's public API is `read_h5ad(path) -> LoadedInput` + `model.assign(&input) ->
AssignmentResult`. v0.1 reuses that as-is. v0.2 can move the Arrow boundary inward.

### Why in-memory return (not write-to-disk)

A binding that wrote a fresh h5ad and made the user `read_h5ad` it back would force
two h5ad round trips per call and forbid the natural Jupyter pattern of "load,
modify, assign, inspect." Returning an in-memory AnnData lets users chain into
their existing workflow without disk I/O.

### Two-layer implementation

**Rust side** (`kaichi-py/src/lib.rs`) — one private `#[pyfunction]`:

```rust
fn _assign_from_h5ad_inmem(
    py: Python,
    h5ad_path: &str,
    model: &str,
    // model kwargs (forwarded selectively per model)
    min_confidence: Option<f32>,
    max_em_iters: Option<u32>,
    // ... a small bounded set; everything else stays at defaults for v0.1
) -> PyResult<(
    PyRecordBatch,         // per-cell assignment columns
    PyArray1<u32>,         // assigned CSR indptr (n_cells + 1)
    PyArray1<u32>,         // assigned CSR indices (nnz)
    PyArray1<u32>,         // counts X CSR data (preserved from input)
    PyArray1<u32>,         // counts X CSR indices
    PyArray1<u32>,         // counts X CSR indptr
    Vec<String>,           // cell barcodes
    Vec<String>,           // guide ids
    String,                // model name (for uns)
    String,                // model params JSON (for uns)
)>
```

The function:
1. `kaichi_core::io::read::read_h5ad(path)` → `LoadedInput`
2. Instantiate the model from `model` string + kwargs
3. `model.assign(&input)` → `AssignmentResult` (RecordBatch + binary CSR)
4. Borrow the count CSR + assigned CSR as numpy arrays via the `numpy` crate
5. Wrap the RecordBatch as `PyRecordBatch` via `pyo3-arrow`
6. Return the tuple

**Python side** (`kaichi-py/python/kaichi/__init__.py`) — the public wrapper:

```python
def assign(h5ad_path: str, model: str = "poisson", **kwargs) -> anndata.AnnData:
    (obs_batch, a_indptr, a_indices,
     x_data, x_indices, x_indptr,
     cell_barcodes, guide_ids,
     model_name, model_params_json) = _rust._assign_from_h5ad_inmem(
        h5ad_path, model, **kwargs)

    n_cells, n_guides = len(cell_barcodes), len(guide_ids)
    X = sp.csr_matrix((x_data, x_indices, x_indptr), shape=(n_cells, n_guides))
    assigned = sp.csr_matrix(
        (np.ones(len(a_indices), dtype=np.uint8), a_indices, a_indptr),
        shape=(n_cells, n_guides),
    )
    obs = obs_batch.to_pandas()
    obs.index = pd.Index(cell_barcodes, name="cell_barcode")

    return anndata.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(guide_ids, name="guide_id")),
        layers={"assigned": assigned},
        uns={"kaichi": {
            "model": model_name,
            "model_params": json.loads(model_params_json),
            "version": kaichi.__version__,
        }},
    )
```

The binary `assigned` layer's `data` array is constructed Python-side (all ones)
rather than shuttled across — it's redundant information that doesn't merit the
Rust→Python handoff.

### Why we're not on the latest pyo3-arrow

`pyo3-arrow` 0.17 (current) tracks `arrow` 58. The kaichi workspace is pinned to
`arrow` 53. Bumping the workspace's arrow version would cascade through
everything — `hdf5-metno`, `nalgebra-sparse`, `arrow-csv` callers in tests, etc.
We stay on `pyo3-arrow` 0.5 (which targets `arrow` 53) for now and revisit when
the workspace next does a coordinated arrow bump.

Authoritative versions live in `kaichi-py/Cargo.toml` and the workspace
`Cargo.toml`, not here.

### Build / dev flow

```sh
# one-time setup
uv venv .venv
uv pip install maturin pytest pyarrow anndata scipy numpy

# build + install the extension into the venv
maturin develop --manifest-path kaichi-py/Cargo.toml

# run Python tests
pytest tests/python/
```

### What v0.1 deliberately omits

- **`scope = "guide" | "batch"` for Gauss** — kwargs forwarded only to the four
  models the user reaches for first (umi, ratio, poisson, neg_binomial). Adding
  the rest is a matter of widening the dispatch.
- **Per-model classes** — model name as a string is enough; classes can wrap
  later without breaking the API.
- **In-memory input (AnnData → Rust)** — Rust still owns h5ad reading. The
  inverse (Python passing an existing AnnData in) is a v0.2 addition that
  requires the `numpy`-crate path for receiving sparse arrays.
- **File output** — no `kaichi.assign_h5ad(in, out, ...)` convenience yet; the
  user can write the returned AnnData themselves with `adata.write_h5ad(out)`.
- **MuData / h5mu** — out of scope until the RNA-modality story is real.

These are all deliberate v0.2+ items; the v0.1 API doesn't preclude any of them.

---

## Target architecture (long-term)

## The contract

```
┌─────────────────────────────────────────────────────────┐
│ Rust core public API                                    │
│                                                         │
│   fn assign(counts:     RecordBatch,                    │
│             guides:     RecordBatch,                    │
│             covariates: RecordBatch,                    │
│             model:      Box<dyn AssignmentModel>,       │
│             params:     ModelParams)                    │
│       -> Result<RecordBatch>                            │
│                                                         │
│ All public types are Arrow record batches.              │
│ The core never sees Python or R types.                  │
└─────────────────────────────────────────────────────────┘
              ▲                          ▲
              │                          │
              │ Arrow C Data Interface   │ Arrow C Data Interface
              │ (PyCapsule protocol)     │ (extendr arrow bridge)
              │                          │
┌─────────────┴────────┐    ┌────────────┴────────────────┐
│ kaichi-py            │    │ kaichi-r                    │
│ - pyo3 + pyo3-arrow  │    │ - extendr + arrow-rs        │
│ - AnnData ↔ Arrow    │    │ - Seurat / SCE ↔ Arrow      │
│ - mudata for writes  │    │ - anndataR for writes (opt) │
└──────────────────────┘    └─────────────────────────────┘
```

The Rust core's public function signatures take and return `RecordBatch` (or
something equivalent like `ArrayRef`). Both bindings call the exact same function
with Arrow data. **Bit-identical inputs produce bit-identical outputs across
bindings** by construction.

---

## Why Arrow at the boundary

Three concrete benefits:

1. **Zero-copy via the C Data Interface.** PyArrow and arrow-rs both implement
   Arrow's C ABI. Data buffers are shared by pointer, not serialized.
2. **Language-neutral schema.** A `RecordBatch` means the same thing in Python,
   R, and Rust. No JSON munging, no Protobuf, no manual struct-packing.
3. **Both ecosystems already use it.** Polars-py, PyArrow, ADBC, scanpy's
   experimental zero-copy paths — Arrow is the de facto interop format in
   modern data tooling.

---

## Python side: kaichi-py

### Library deps
- `pyo3` — Python ↔ Rust glue
- `pyo3-arrow` (or `arro3-core`) — Arrow C Data Interface bridge
- `anndata`, `mudata` — file I/O for in-memory paths (optional read, mandatory write)
- numpy / scipy.sparse — only for converting AnnData internals to Arrow

### From AnnData to Arrow

When the user passes an in-memory AnnData (rather than a file path):

```python
def _adata_to_arrow_inputs(adata):
    # Cell barcodes
    barcodes = pa.array(adata.obs_names.to_numpy(), type=pa.large_string())

    # Sparse X as three Arrow arrays (CSR)
    X = adata.X.tocsr() if not sp.isspmatrix_csr(adata.X) else adata.X
    counts = pa.RecordBatch.from_arrays([
        pa.array(X.data),
        pa.array(X.indices),
        pa.array(X.indptr),
    ], names=["data", "indices", "indptr"])

    # var: guide IDs plus optional guide-library metadata
    var = pa.Table.from_pandas(
        adata.var[["guide_id", "target_gene", "sequence", ...]]
    )

    # Covariates (per-cell batch, total_counts) from .obs
    covariates = pa.Table.from_pandas(
        adata.obs[["batch", "total_counts"]]
    )

    return barcodes, counts, var, covariates
```

Each `pa.RecordBatch` / `pa.Array` exposes `__arrow_c_array__` /
`__arrow_c_stream__`; pyo3-arrow on the Rust side consumes those as
PyCapsules without copying.

### From Arrow result to MuData

The Rust core returns one `RecordBatch` for the crispr modality's `.obs`
(per-cell assignment) and the (cells × guides) sparse `.X` matrix. The Python
binding reconstructs an `AnnData`:

```python
def _arrow_to_anndata(obs_batch, X_csr, var_table):
    return anndata.AnnData(
        X = sp.csr_matrix((X_csr["data"], X_csr["indices"], X_csr["indptr"])),
        obs = obs_batch.to_pandas(),
        var = var_table.to_pandas(),
        uns = {"kaichi": _build_provenance(...)},
    )
```

Then either:
- Wrap it into a MuData with the (optionally provided) RNA modality and write
  with `mdata.write_h5mu()`.
- If no RNA was provided and the user wants H5AD, write directly.

### Sparse matrix handoff: three arrays, one RecordBatch

Arrow doesn't have a single first-class CSR/CSC type that all consumers handle
cleanly. We pass the three CSR arrays as named fields of a RecordBatch:

```
schema: data    primitive (Float32 or UInt32 depending on context)
        indices primitive (UInt32 or UInt64)
        indptr  primitive (UInt32 or UInt64)
```

Rust reconstructs a view (`sprs::CsMat` or a thin custom struct) over these.
Polars and DuckDB use the same convention.

---

## R side: kaichi-r

### Library deps
- `extendr-api` — R ↔ Rust glue
- `arrow` (R package) — Arrow record batch / array support in R
- `Seurat` — primary R object construction
- `SingleCellExperiment` (Bioconductor) — alternative when `format = "sce"`
- `anndataR` — optional, only if reading H5AD from R is needed

### Arrow handoff

extendr supports passing Arrow data via the `nanoarrow`-compatible R bindings.
Conceptually identical to the Python case: PyCapsule on Python side becomes
the `nanoarrow_array` / `nanoarrow_schema` on R side, but both implement
Arrow's C Data Interface ABI under the hood.

R-side construction is the inverse of Python-side:

```r
.assign_guides_impl <- function(counts, guides, covariates, model, params) {
    result <- .Call("kaichi_assign", counts, guides, covariates, model, params)
    # result is a list with: obs_batch, X_csr (data/indices/indptr), var_table
    .build_seurat(result)  # or .build_sce(result) if format = "sce"
}
```

### Seurat construction

The default output object. Guide modality lives as an Assay (Seurat v5: an
Assay5):

```r
.build_seurat <- function(result, rna = NULL) {
    crispr_counts <- sparseMatrix(
        i = result$X_csr$indices + 1,
        p = result$X_csr$indptr,
        x = result$X_csr$data,
        dims = c(nrow(result$var), nrow(result$obs)),
    )
    crispr_assay <- CreateAssayObject(counts = crispr_counts)

    if (is.null(rna)) {
        # guide-only object
        obj <- CreateSeuratObject(counts = crispr_counts, assay = "crispr")
    } else {
        # rna object passed in; attach crispr assay
        obj <- rna
        obj[["crispr"]] <- crispr_assay
    }
    obj@meta.data <- cbind(obj@meta.data, result$obs %>% to_meta_columns())
    obj
}
```

`format = "sce"` produces a `SingleCellExperiment` with `altExp("crispr")`
instead.

---

## Cross-language equivalence guarantee

Because the Rust core sees only Arrow record batches, and the bindings only
do format conversion (no math), the cross-language equivalence test in
[validation.md](validation.md) reduces to:

1. Build a fixed Arrow input fixture.
2. Pass it to kaichi-py and kaichi-r.
3. Compare the returned `RecordBatch` byte-for-byte (or float-equal within
   `1e-6` for EM models) **before** any AnnData / Seurat construction.

If this test passes, downstream language-specific differences (Pandas
vs base R DataFrame, sparse matrix conventions) are purely cosmetic.

---

## Implementation notes

### Sparse matrix dtype
- Counts are non-negative integers → `UInt32` for `.X.data` is sufficient even
  for very deep guide sequencing.
- Indices and indptr: `UInt32` for typical sizes; promote to `UInt64` if a
  single matrix exceeds ~4G nnz (unlikely for guide data).

### Categorical columns
- `guide_id`, `target_gene`, `batch` are categorical. `target_gene` is nullable
  when guide metadata is absent. Arrow `DictionaryArray` preserves the encoding
  through the boundary; both pyo3-arrow and the R arrow package handle it natively.
  Avoid materializing as raw strings unless you need to.

### Memory layout in Rust
- The compute kernel typically wants the (cells × guides) sparse matrix in
  CSR with cells as rows.
- The covariate frame is a small dense `ndarray::Array2` indexed by cell.
- Per-guide work iterates columns of the CSR; rayon parallelizes that loop.

### When in-memory inputs come in WITHOUT an existing AnnData

If a user calls `kaichi.assign_guides(counts="cellranger_dir/")`
with no `input=` argument, no AnnData ever exists on the Python side until the
end. The flow is:
1. Python binding hands `counts_dir` (a string) to Rust.
2. Rust reads the MTX in-process and produces Arrow batches internally.
3. Rust runs the model, returns Arrow result.
4. Python binding constructs AnnData / MuData from the result.

Same flow for R. The disk read happens in Rust regardless of language.

If `guide_library="guides.tsv"` is supplied, the binding passes that path through to
the Rust core for validation and metadata enrichment. If omitted, the result is still
constructed from guide IDs present in the count input.

---

## Open verification items

Before scaffolding the bindings:

- Confirm pyo3-arrow's API stability (it's been moving; arro3-core is an
  alternative).
- Confirm extendr's Arrow handoff story — `extendr-api` supports nanoarrow
  through `arrow::FFI_ArrowSchema` / `FFI_ArrowArray` exchange, but the API
  ergonomics vary.
- Verify Seurat v5's `Assay5` is the right slot vs the older `Assay` class
  for the crispr modality on R side.

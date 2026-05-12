# kaichi — Binding Interop

How data crosses the Rust ↔ Python and Rust ↔ R boundaries. The contract here is
what makes "one Rust core, two bindings, identical results" actually work.

---

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

    # var (guide library): subset of columns we care about
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
- `guide_id`, `target_gene`, `batch` are categorical. Arrow `DictionaryArray`
  preserves the encoding through the boundary; both pyo3-arrow and the R
  arrow package handle it natively. Avoid materializing as raw strings unless
  you need to.

### Memory layout in Rust
- The compute kernel typically wants the (cells × guides) sparse matrix in
  CSR with cells as rows.
- The covariate frame is a small dense `ndarray::Array2` indexed by cell.
- Per-guide work iterates columns of the CSR; rayon parallelizes that loop.

### When in-memory inputs come in WITHOUT an existing AnnData

If a user calls `kaichi.assign_guides(counts="cellranger_dir/", guide_library="guides.tsv")`
with no `input=` argument, no AnnData ever exists on the Python side until the
end. The flow is:
1. Python binding hands `counts_dir` (a string) to Rust.
2. Rust reads the MTX in-process and produces Arrow batches internally.
3. Rust runs the model, returns Arrow result.
4. Python binding constructs AnnData / MuData from the result.

Same flow for R. The disk read happens in Rust regardless of language.

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

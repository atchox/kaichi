# kaichi — Storage and Encoding

This doc pins down how kaichi reads and writes guide-count and assignment data on
disk. It is the authoritative reference for the file formats kaichi accepts and
produces, the HDF5 group layouts kaichi targets, and the spec compliance the
implementation must hit.

---

## Core principle

kaichi-core is **Python-free for all disk I/O**. The CLI is a standalone Rust
binary; the Python and R bindings only do format work for in-memory object
construction. On-disk read/write happens entirely in Rust.

kaichi reads and writes the AnnData and MuData on-disk specs **directly via the
`hdf5-metno` crate**. We own the spec interpretation in `kaichi-core/src/io/`.

---

## Two layers of the I/O stack

```
┌──────────────────────────────────────────────────────┐
│ Layer 2: Format-specific readers/writers             │
│   - MEX directory reader (.mtx.gz + .tsv.gz)         │
│   - H5AD reader/writer (AnnData spec)                │
│   - H5MU reader/writer (MuData spec)                 │
│   - CSV writer (flat assignment table)               │
└──────────────────────────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│ Layer 1: Raw HDF5 + gzip + line-oriented I/O         │
│   - hdf5-metno for HDF5 primitives + H5Ocopy         │
│   - flate2 for gzipped MTX/TSV                       │
│   - std::io for CSV and uncompressed text            │
└──────────────────────────────────────────────────────┘
```

---

## Input formats

kaichi accepts three input forms for `--counts`. The format is detected from the
path: directory → MEX; `.h5ad` extension → H5AD; `.h5mu` extension → H5MU. An
explicit `--input-format` flag overrides detection.

### 1. MEX — Cell Ranger feature-barcode matrix directory

The directory layout produced by `cellranger count` / `cellranger multi`:

```
filtered_feature_bc_matrix/   (or raw_feature_bc_matrix/)
├── matrix.mtx.gz       Matrix Market triplet format, gzipped
├── barcodes.tsv.gz     one cell barcode per line, gzipped
└── features.tsv.gz     feature_id<TAB>feature_name<TAB>feature_type, gzipped
```

**`matrix.mtx.gz`** (Matrix Market `coordinate integer general`):

```
%%MatrixMarket matrix coordinate integer general
%
<n_features> <n_barcodes> <nnz>
<feature_idx_1based> <barcode_idx_1based> <count>
<feature_idx_1based> <barcode_idx_1based> <count>
...
```

Cell Ranger writes **features as rows, barcodes as columns** (the convention is
swapped from scanpy's cells-as-rows). kaichi transposes during read.

**`features.tsv.gz`** mixes feature types. For Perturb-seq runs the column
values are `Gene Expression` and `CRISPR Guide Capture`. kaichi partitions
features by type and builds **two** sparse matrices in a single pass over
`matrix.mtx.gz`:

- guide matrix (cells × `CRISPR Guide Capture` features) — used for assignment
- RNA matrix (cells × `Gene Expression` features) — passed through to output

If the directory contains no Gene Expression rows, only the guide matrix is
built and the output is guide-only H5AD.

**Read procedure:**

1. Open `features.tsv.gz` with `flate2::read::GzDecoder`. Partition rows into
   two index sets keyed by `feature_type`: `crispr_features` and
   `rna_features`. For each kept row, record the original 1-based
   `mtx_feature_idx`, the `feature_id`, and the `feature_name`.
2. Open `barcodes.tsv.gz`; collect cell barcodes in file order.
3. Open `matrix.mtx.gz`; skip the comment header; parse the dimensions line;
   stream triplets. For each `(feature_idx, barcode_idx, count)`, route into
   the COO buffer of the matching feature set (or skip if `feature_idx`
   belongs to neither, defensively).
4. Sort each COO by `(barcode_idx, kept_pos)`; build a CSR per feature set
   with `n_cells` rows. Cast `data` to `u32`.

`features.tsv.gz` columns are also carried into the respective `var`
DataFrames: `feature_id` becomes `_index` (`guide_id` or `gene_id`),
`feature_name` becomes a `gene_symbol` / `guide_name` column.

### 2. H5AD — guides only

When the input is `.h5ad`, kaichi treats it as a **guide-only AnnData**:

- `obs_names` = cell barcodes
- `var_names` = guide IDs
- `X` = sparse UMI count matrix, cells × guides

This is the format scanpy / muon produce when a user has already extracted the
`crispr` modality. kaichi reads `X`, `obs/_index`, `var/_index`. Optional `var`
columns (target_gene, sequence, etc.) are read if present; missing columns are
treated as null.

If the input H5AD has more than one feature type encoded in `var`, kaichi does
**not** filter — the assumption is that an H5AD is already curated. To process
a mixed-feature dataset, use MEX or H5MU input instead.

### 3. H5MU — multi-modal

When the input is `.h5mu`, kaichi expects a MuData container with at least a
`crispr` modality and optionally an `rna` modality:

```
file.h5mu
├── attrs: encoding-type = "MuData", mod-order = [...]
└── mod/
    ├── rna/      (AnnData — gene expression; opaque to kaichi)
    └── crispr/   (AnnData — guide counts; read for assignment)
```

**What kaichi reads:** the `crispr` modality only — `mod/crispr/obs/_index`,
`mod/crispr/var/_index`, `mod/crispr/X`. Treated identically to the H5AD case
above, just rooted at the `/mod/crispr/` group instead of the file root.

**What kaichi does NOT parse:** the `rna` modality. It stays unread on disk and
is passed through to the output via `H5Ocopy` (see structural pass-through
below). This is intentional — kaichi has no opinion on RNA encoding, and not
parsing it means we are forward-compatible with whatever encoding the upstream
producer used (Compressed Sparse Matrix, Dense, BPCells, etc.).

If `crispr` is absent, kaichi errors out. If extra modalities (`atac`, `adt`,
etc.) are present, they are passed through to the output unmodified.

---

## Output formats

kaichi produces three output forms. Like inputs, the format is detected from
the `--output` path extension; an explicit `--output-format` overrides.

### 1. H5AD — guide-only AnnData (default when no RNA companion)

```
file.h5ad
├── attrs: encoding-type = "anndata", encoding-version = "0.1.0"
├── X/                          sparse CSR UInt32 (cells × guides) — raw UMI counts
├── layers/
│   └── assigned/               sparse CSR UInt8 — binary call matrix
├── obs/                        per-cell assignment DataFrame (see data-model.md)
│   ├── cell_barcode            (string-array, _index)
│   ├── guide_identity          (string-array)
│   ├── n_guides_assigned       (int32)
│   ├── assignment_confidence   (float32)
│   ├── is_unassigned           (bool)
│   └── is_multi_infected       (bool)
├── var/
│   ├── guide_id                (string-array, _index)
│   └── (optional library metadata if --guide-library was provided)
└── uns/
    └── kaichi/                 provenance group (model, params, version, run_stats)
```

Triggered when:
- Input is MEX with no Gene Expression rows (CRISPR-only library)
- Input is H5AD (always guide-only by convention)
- User explicitly passes `--output out.h5ad`, even when RNA is available
  (kaichi warns that RNA is being dropped)

### 2. H5MU — multi-modal (default when RNA is available)

```
file.h5mu
├── attrs: encoding-type = "MuData", encoding-version = "0.1.0"
│          mod-order = ["rna", "crispr", ...]
└── mod/
    ├── rna/      (H5Ocopy'd from input — unchanged)
    ├── crispr/   (AnnData group — same layout as standalone H5AD above)
    └── <other>/  (H5Ocopy'd from input if any extra modalities were present)
```

Triggered when:
- Input is H5MU (RNA modality and any other modalities pass through via H5Ocopy)
- Input is MEX with both Gene Expression and CRISPR Guide Capture rows
  (RNA modality is built fresh from the GEX rows, not H5Ocopy'd)

The `crispr` modality is written from scratch using kaichi's AnnData spec
writer. For H5MU input, other modalities (including `rna`) are byte-copied via
`H5Ocopy` without semantic interpretation — forward-compatible and preserves
compression filters, internal references, and attributes set by upstream
tools. For MEX input, the `rna` modality is built from the GEX rows of the
same `matrix.mtx.gz` and written with the same spec writer as `crispr`.

### 3. CSV — flat assignment table (for pipeline compat)

```
cell,gRNA,UMI_counts
AAACCTGAGAAACCAT-1,sgCDKN1A_1,42
AAACCTGAGAAACCAT-1,sgTP53_2,3
...
```

One row per **assigned** cell-guide pair (cells with `is_unassigned=true` are
omitted; multi-infected cells emit one row per detected guide). No metadata,
no provenance, no `.var` enrichment. Used for:

- Snakemake / Nextflow pipelines that already have downstream parsers
- Equivalence comparisons against crispat / SCEPTRE outputs
- Quick visual inspection during development

Triggered when:
- `.csv` (or `.tsv`) extension on `--output`

CSV output is a **lossy** format relative to the H5AD / H5MU outputs. kaichi
emits a warning at write time noting that per-cell counts, provenance, and
multi-infection metadata are discarded.

---

## Input × output compatibility matrix

The shape of `--counts` decides what's available in the output. kaichi has no
flag for attaching extra modalities; whatever is in `--counts`, kaichi
preserves.

| Input | RNA present? | Default output | Allowed outputs |
|---|---|---|---|
| MEX (CRISPR + GEX) | yes | H5MU | H5MU, H5AD, CSV |
| MEX (CRISPR only) | no | H5AD | H5AD, CSV |
| H5AD (guides only) | no | H5AD | H5AD, CSV |
| H5MU (rna + crispr) | yes | H5MU | H5MU, H5AD, CSV |

**Notes:**

- `H5AD` is allowed as an output even when RNA is present in the input; kaichi
  writes the guide-only AnnData and warns that RNA was dropped.
- `H5MU` is **disallowed** when no RNA source exists — kaichi cannot
  fabricate an RNA modality and errors out at output-format selection.

---

## AnnData on-disk spec (the slice kaichi implements)

The authoritative spec is at
<https://anndata.readthedocs.io/en/latest/fileformat-prose.html>. kaichi
implements the encoding details below; everything else in the spec is either
out of scope or read-only pass-through via `H5Ocopy`.

### File-level attributes

```
encoding-type     = "anndata"        (str scalar, UTF-8)
encoding-version  = "0.1.0"          (str scalar)
```

### Sparse matrix (`X`, `layers/<name>`, `obsp/<name>`, …)

CSR encoded as a group with three datasets and group-level attributes:

```
X/                                     (group)
├── attrs: encoding-type     = "csr_matrix"
│          encoding-version  = "0.1.0"
│          shape             = [n_rows, n_cols]  (int64[2])
├── data    (1-D dataset; uint32 for UMI counts, uint8 for binary layers)
├── indices (1-D int32 dataset, length = nnz)
└── indptr  (1-D int32 dataset, length = n_rows + 1)
```

`data` is gzip-compressed (level 4); `indices` and `indptr` are compressed if
length exceeds 64KB, uncompressed otherwise. Chunking is row-major at ~1MB.
Promote `indices` / `indptr` to `int64` only if a single matrix exceeds 2G nnz
(unlikely for guide-count sizes).

### DataFrame (`obs`, `var`)

Group with one dataset per column plus group-level metadata:

```
obs/
├── attrs: encoding-type     = "dataframe"
│          encoding-version  = "0.2.0"
│          _index            = "cell_barcode"
│          column-order      = ["guide_identity", "n_guides_assigned", ...]
├── cell_barcode        (string-array  — the index)
├── guide_identity      (string-array)
├── n_guides_assigned   (int32-array)
└── ...
```

Each column is a dataset (or sub-group for categorical) with its own
`encoding-type` / `encoding-version` attributes:

- `string-array` (0.2.0) — variable-length UTF-8 strings
- `array` (0.2.0) — fixed-width numeric / bool
- `categorical` (0.2.0) — see below

### Categorical column (sub-group)

```
guide_id/
├── attrs: encoding-type     = "categorical"
│          encoding-version  = "0.2.0"
│          ordered           = false
├── codes        (int8 / int16 / int32, -1 for null)
└── categories   (string-array — the dictionary values)
```

Matches Arrow `DictionaryArray` 1:1 at the buffer level. Codes width is
selected from the cardinality of `categories`: <128 → i8, <32k → i16, else i32.

### `uns`

Group of arbitrary nested values. kaichi writes:

- string scalars (model name, kaichi version, timestamp)
- numeric scalars (cell counts)
- nested groups for `model_params` and `run_stats`

Implemented via a small recursive `write_uns(group, &serde_json::Value)`.

### Index storage

In modern AnnData (encoding-version ≥ 0.2.0), the row index lives as the
`_index` column of `obs` / `var`, not as a top-level `obs_names` / `var_names`
dataset. kaichi writes only the modern form.

---

## MuData on-disk spec

Small wrapper around AnnData groups. kaichi writes:

```
file.h5mu
├── attrs:
│     encoding-type     = "MuData"
│     encoding-version  = "0.1.0"
│     mod-order         = ["rna", "crispr", ...]   (string array, ordered)
└── mod/
    ├── rna/      (AnnData group — H5Ocopy'd from input or absent)
    ├── crispr/   (AnnData group — written by kaichi's AnnData writer)
    └── <other>/  (H5Ocopy'd from input if present)
```

The AnnData spec writer takes a target `Group`, not a file path. Writing the
`crispr` modality to `/mod/crispr/` is the same code as writing a standalone
H5AD to `/` — just a different parent group.

The exact attribute names and encoding versions kaichi pins to live as
constants in `kaichi-core/src/io/spec.rs`.

---

## Structural pass-through (H5Ocopy)

When the user provides an existing H5MU and asks kaichi to attach / replace the
`crispr` modality, kaichi does **not parse the unchanged content**. It copies
it byte-for-byte.

**Mechanism.** HDF5's `H5Ocopy` (C API) copies any object (group, dataset, with
attributes and internal references) to another location, potentially in another
file. `hdf5-metno` exposes this.

**When kaichi uses it:**

| Input | Output `crispr` from | Pass-through |
|---|---|---|
| H5MU with rna + crispr | recompute from input's crispr counts | Copy `/mod/rna/` over; replace `/mod/crispr/` |
| H5MU with rna + crispr + atac | recompute crispr | Copy `/mod/rna/` and `/mod/atac/`; replace `/mod/crispr/` |

For MEX input with Gene Expression rows, the `rna` modality is **not**
H5Ocopy'd — it's built from scratch by the MEX reader and written via the
AnnData spec writer (same path as `crispr`). H5Ocopy applies only when the
source is itself an HDF5 file with a pre-encoded modality.

**Why this is safe:**
- Attributes preserved
- Compression filters preserved (if the destination file has the filter
  available — flag mismatches as errors)
- Internal references preserved (categorical encoding uses these)
- Forward-compatible with AnnData spec additions kaichi has never seen

**What's NOT safe:**
- External links (objects in other files referenced by symbolic link). AnnData
  and MuData don't use these in practice; verify in a test.
- Exotic compression filters (custom BLOSC plugins) — propagate filter
  availability as a clear error rather than silent failure.

---

## Encoding decisions pinned at v0

| Decision | Choice | Reason |
|---|---|---|
| Sparse matrix encoding | CSR | AnnData spec default; matches Cell Ranger output |
| Sparse `data` dtype | `UInt32` for counts, `UInt8` for binary layers | Compact, sufficient |
| CSR `indices` / `indptr` dtype | `Int32` | Sufficient for < 2G nnz |
| Categorical encoding | `DictionaryArray` ↔ AnnData `categorical`, codes width = `i8` / `i16` / `i32` by cardinality | Spec-compliant; no intermediate |
| String encoding | UTF-8, variable-length | Modern AnnData default |
| Compression | gzip level 4 on `data` datasets; none on small metadata | h5py readable without external plugins |
| Chunking | row-major, ~1MB chunks for `X` | Reasonable for cell-wise access |
| AnnData `encoding-version` | "0.1.0" for sparse, "0.2.0" for dataframe / categorical | Match upstream Python anndata defaults |
| H5MU `encoding-version` | "0.1.0"; bump explicitly | Forward compatibility |
| `_index` column name | `cell_barcode` for `.obs`, `guide_id` for `.var` | Matches kaichi's data model |
| MEX feature filter | `feature_type == "CRISPR Guide Capture"` | Cell Ranger convention; case-sensitive match |

---

## Spec compliance and verification

Because kaichi owns the encoding, kaichi owns the compliance burden.
Two-pronged:

1. **Round-trip tests inside the repo.** Write an H5AD with kaichi, read it
   back with kaichi, assert equality on every field. Same for H5MU. Located
   in `kaichi-core/tests/write_h5ad.rs` (and `write_h5mu.rs` once added).
2. **Python interop tests.** A test that shells out to
   `python -c "import anndata; ad.read_h5ad(...)"` (and `muon.read_h5mu` for
   the H5MU case) and asserts the file loads without warnings and the
   `.obs` / `.var` / `.X` shapes / dtypes match expectations. Gated on Python
   availability; skipped otherwise. This is the canonical compliance check
   for the spec kaichi does not control.

For MEX, the round-trip check is "kaichi reads `cellranger`-produced output
and matches the expected `n_cells × n_guides` with correct UMI counts on a
known fixture."

---

## In-memory bindings — not this doc's concern

When the user passes an in-memory AnnData (Python) or Seurat object (R), no
HDF5 is involved. The language binding extracts what kaichi needs as Arrow and
the Rust core returns Arrow. The on-disk path described here is bypassed.

See [binding-interop.md](binding-interop.md) for that flow and
[memory-layout.md](memory-layout.md) for how data is represented inside the
Rust core.

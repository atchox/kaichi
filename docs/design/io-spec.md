# kaichi — I/O Specification

## Inputs

### 1. Guide count data (required, one of)

#### a. Cell Ranger feature-barcode matrix directory
Standard Cell Ranger `filtered_feature_bc_matrix/` or `raw_feature_bc_matrix/`:
```
matrix.mtx.gz       sparse count matrix (cells × features)
barcodes.tsv.gz     cell barcodes
features.tsv.gz     feature metadata (id, name, feature_type)
```
kaichi reads only rows where `feature_type == "CRISPR Guide Capture"`.
Gene Expression rows are ignored — RNA comes from the caller.

#### b. Cell Ranger `molecule_info.h5`
Provides UMI-level records before cell/UMI filtering. Enables kaichi to apply its own
deduplication strategy rather than accepting Cell Ranger's defaults.
Preferred when the assignment model needs raw UMI counts (e.g., mixture models).

### 2. Guide library reference (optional)
Counts are sufficient to assign cells to guide IDs. A guide library is optional
metadata used to validate guide IDs and enrich outputs with biological annotations.

When provided, it is a TSV with at minimum:

```
guide_id        target_gene     sequence
sgCDKN1A_1      CDKN1A          ACGTACGTACGTACGT
sgCDKN1A_2      CDKN1A          TTGGCAATGCTAGCAT
sgNTC_1         non-targeting   AAAAAAAAAAAAAAAA
```

Additional optional columns (included in `.var` of the output CRISPR modality if present):
`chromosome`, `cut_site`, `strand`, `on_target_score`

If no guide library is provided, kaichi still emits a valid result with guide IDs from
the count matrix. In that mode `.var` contains a minimal guide table and annotation
fields such as `target_gene` and `sequence` are null or absent.

### 3. Existing single-cell object (optional)

kaichi can attach guide assignment results to an existing object rather than producing a
standalone output:

| Input type | Behavior |
|---|---|
| H5AD (AnnData) | Wrap in MuData as `mod['rna']`; add guide calls as `mod['crispr']` |
| H5MU (MuData) | Add or overwrite `mod['crispr']`; leave all other modalities untouched |

Cell barcodes are inner-joined between the guide data and the existing object.
Cells present in one but not the other are handled per `barcode_join` option:
- `inner` (default) — keep only shared barcodes
- `left` — keep all cells from the existing object; unassigned cells get null guide calls
- `outer` — keep all barcodes from both; fill missing values with nulls

---

## Outputs

### Primary: H5MU

```
MuData (.h5mu)
├── mod['rna']     AnnData  — gene expression (passthrough if provided, absent otherwise)
└── mod['crispr']  AnnData  — guide assignment results
    ├── X                   sparse UInt32 matrix  (cells × guides, raw UMI counts)
    ├── obs                 per-cell assignment (see Data Model)
    ├── var                 guide IDs plus optional guide library metadata
    └── uns['kaichi']       run metadata: model used, parameters, kaichi version
```

### Fallback: H5AD (guide-only)
When no RNA object is provided and the caller wants a simpler file:
A single AnnData where `.X` = guide counts, `.obs` = guide calls, and `.var` =
guide IDs plus optional guide library metadata.

### In-memory (binding return value)

Python binding returns a `MuData` object (never writes a file unless asked).
R binding returns a Seurat object by default, with the guide modality as an assay
named `"crispr"`. A `format = "sce"` option returns a `SingleCellExperiment` with the
guide modality in `altExp("crispr")` for Bioconductor users.

The Rust core returns an Arrow `RecordBatch` per modality; the bindings assemble the
language-native objects from those batches.

---

## Python API sketch

```python
import kaichi

# Minimal — from Cell Ranger directory
result = kaichi.assign_guides(
    counts="path/to/filtered_feature_bc_matrix/",
    model="max",                # see assignment-models.md
)

# Attach to existing RNA object, fit a per-guide NB mixture
result = kaichi.assign_guides(
    counts="molecule_info.h5",
    guide_library="guides.tsv",  # optional metadata / validation
    input="rna.h5ad",           # AnnData or MuData; path or in-memory object
    model="neg_binomial",
    barcode_join="inner",
)

# Write
result.write("out.h5mu")
```

## R API sketch

```r
library(kaichi)

# Default — returns a Seurat object; guide modality as assay "crispr"
obj <- assign_guides(
  counts = "filtered_feature_bc_matrix/",
  model = "max",
  input = "rna.rds"     # optional; path, Seurat object, or H5AD/H5MU path
)

# Bioconductor users — returns a SingleCellExperiment with altExp("crispr")
sce <- assign_guides(
  counts = "filtered_feature_bc_matrix/",
  guide_library = "guides.tsv", # optional metadata / validation
  model = "neg_binomial",
  format = "sce"
)
```

The R binding accepts the same input types as the Python binding:
H5AD or H5MU paths are read via Rust HDF5; Seurat/SCE objects in memory are converted
to Arrow record batches before being passed into the core.

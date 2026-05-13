# kaichi — I/O Specification

This doc defines the user-facing inputs and outputs of kaichi. For the on-disk
file layouts kaichi reads and writes, see
[storage-encoding.md](storage-encoding.md).

---

## Inputs

### 1. Guide count data (required) — `--counts <PATH>`

Three accepted forms. Format detection from the path:

| `<PATH>` | Detected as |
|---|---|
| directory containing `matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz` | MEX |
| file ending in `.h5ad` | H5AD (guides only) |
| file ending in `.h5mu` | H5MU (multi-modal) |

An explicit `--input-format {mex,h5ad,h5mu}` overrides detection.

#### 1a. MEX — Cell Ranger feature-barcode matrix directory

Either `filtered_feature_bc_matrix/` or `raw_feature_bc_matrix/`. kaichi reads
**both** feature types:

- Rows with `feature_type == "CRISPR Guide Capture"` build the guide count
  matrix used for assignment (output `mod['crispr'].X`).
- Rows with `feature_type == "Gene Expression"` (if any) build an RNA count
  matrix passed through to the output as `mod['rna'].X` with `gene_id` /
  `gene_symbol` carried into `mod['rna'].var` from `features.tsv.gz`.

Both matrices share the same cell axis from `barcodes.tsv.gz` by construction.

If the MEX directory has no Gene Expression rows (CRISPR-only library), kaichi
produces a guide-only output.

#### 1b. H5AD — guides only

An AnnData file whose `.X` is the cells × guides UMI count matrix. kaichi
treats it as **already curated** — no feature-type filtering is performed. Use
this form when you have extracted the `crispr` modality from a multi-modal
object in advance.

Optional `.var` columns (`target_gene`, `sequence`, etc.) are preserved if
present; missing columns are treated as null.

#### 1c. H5MU — RNA + CRISPR

A MuData file with at least a `crispr` modality and optionally an `rna`
modality. kaichi reads `mod['crispr']` exactly as if it were a guide-only H5AD
(case 1b). The `rna` modality is **not parsed** — it is byte-copied to the
output via `H5Ocopy`, preserving every attribute and encoding choice the
upstream producer made.

If `crispr` is absent, kaichi errors. Extra modalities (`atac`, `adt`, …) are
passed through to the output unmodified.

### 2. Guide library reference (optional) — `--guide-library <TSV>`

Counts alone are sufficient to assign cells to guide IDs. A guide library is
optional metadata used to validate guide IDs and enrich the output `.var`.

Minimum schema:

```
guide_id        target_gene     sequence
sgCDKN1A_1      CDKN1A          ACGTACGTACGTACGT
sgCDKN1A_2      CDKN1A          TTGGCAATGCTAGCAT
sgNTC_1         non-targeting   AAAAAAAAAAAAAAAA
```

Optional columns included in `.var` if present: `chromosome`, `cut_site`,
`strand`, `on_target_score`.

If `--guide-library` is omitted, `.var` contains only `guide_id`; annotation
fields are null or absent.

### 3. No separate RNA input

kaichi does **not** accept a separate RNA file. Every modality kaichi produces
in the output comes from `--counts`:

- MEX with both feature types → both modalities
- H5MU with rna + crispr → both modalities
- MEX with CRISPR-only / guide-only H5AD → guides only

If you have RNA in one file and guides in another, assemble them into an H5MU
upstream (e.g., with `muon` in Python or `scx convert`) before passing to
kaichi. This keeps kaichi's contract narrow: one input path in, one output
path out.

---

## Outputs

### Output format selection

Three output forms. Format detection from the `--output` path extension:

| Extension | Output |
|---|---|
| `.h5ad` | H5AD — guide-only AnnData |
| `.h5mu` | H5MU — multi-modal (rna passthrough + crispr from kaichi) |
| `.csv` or `.tsv` | Flat CSV — `(cell, gRNA, UMI_counts)` |

An explicit `--output-format` overrides. If `--output` is omitted, kaichi
writes `<counts_stem>.h5ad` (or `.h5mu` if an RNA source exists) alongside
the input.

### 1. H5AD — single-modality AnnData

```
H5AD (.h5ad)
├── X                       sparse UInt32 cells × guides (raw UMI counts)
├── layers["assigned"]      sparse UInt8  cells × guides (binary call matrix)
├── obs                     per-cell assignment (see data-model.md)
├── var                     guide IDs plus optional guide-library metadata
└── uns["kaichi"]           run provenance (model, parameters, version)
```

Use when no RNA source is available, or when downstream tooling only consumes
single-modality AnnData.

### 2. H5MU — multi-modal MuData

```
H5MU (.h5mu)
├── mod["rna"]              gene expression (passthrough from --input or H5MU --counts)
├── mod["crispr"]           guide assignment results — same shape as standalone H5AD above
└── mod["<other>"]          any additional modalities from the H5MU input, passed through unchanged
```

Use when RNA is available. kaichi's contract is **the rna modality is byte
identical to the source** — kaichi never re-encodes RNA data.

### 3. CSV — flat assignment table

```
cell,gRNA,UMI_counts
AAACCTGAGAAACCAT-1,sgCDKN1A_1,42
AAACCTGAGAAACCAT-1,sgTP53_2,3
...
```

One row per **assigned** cell-guide pair. Cells marked `is_unassigned` are
omitted; multi-infected cells emit one row per detected guide. **Lossy** —
discards per-cell counts of unassigned guides, provenance, multi-infection
flags, and confidence scores. Useful for:

- Snakemake / Nextflow pipelines with existing parsers
- Equivalence comparisons against crispat / SCEPTRE
- Quick visual inspection

kaichi warns at write time that metadata is being dropped.

### 4. In-memory (binding return value)

Python binding returns a `MuData` object when RNA is present, otherwise an
`AnnData`. Never writes a file unless asked.

R binding returns a Seurat object by default with the guide modality as an
assay named `"crispr"`. A `format = "sce"` option returns a
`SingleCellExperiment` with the guide modality in `altExp("crispr")`.

The Rust core returns Arrow `RecordBatch`es per modality; the bindings
assemble the language-native objects from those batches.

---

## Input × output compatibility matrix

The shape of `--counts` decides what's available in the output. There is no
flag that adds modalities; whatever is in `--counts`, kaichi preserves.

| `--counts` | RNA present? | Default `--output` | Permitted `--output` |
|---|---|---|---|
| MEX (CRISPR + GEX) | yes | `.h5mu` | `.h5mu`, `.h5ad`, `.csv` |
| MEX (CRISPR only) | no | `.h5ad` | `.h5ad`, `.csv` |
| H5AD (guides only) | no | `.h5ad` | `.h5ad`, `.csv` |
| H5MU (rna + crispr) | yes | `.h5mu` | `.h5mu`, `.h5ad`, `.csv` |

Resolution rules:

- `--output *.h5mu` is **rejected at parse time** when `--counts` has no RNA
  (CRISPR-only MEX or guide-only H5AD). kaichi cannot fabricate an RNA modality.
- `--output *.h5ad` is permitted when RNA is available; kaichi writes the
  guide-only AnnData and **warns** that RNA was dropped.
- `--output *.csv` always emits a warning that the output is lossy.

---

## Python API sketch

```python
import kaichi

# MEX in (Perturb-seq — has both GEX and CRISPR) → MuData out
result = kaichi.assign_guides(
    counts="path/to/filtered_feature_bc_matrix/",
    model="max",
)
result.write("out.h5mu")

# MEX in (CRISPR-only library — no GEX rows) → AnnData out
result = kaichi.assign_guides(
    counts="crispr_only_mtx/",
    model="max",
)
result.write("out.h5ad")

# H5MU in (rna + crispr) → H5MU out (rna passes through)
result = kaichi.assign_guides(
    counts="combined.h5mu",
    guide_library="guides.tsv",
    model="poisson_gauss",
)
result.write("out.h5mu")
```

## R API sketch

```r
library(kaichi)

# MEX (Perturb-seq) → Seurat object with rna + crispr assays
obj <- assign_guides(
  counts = "filtered_feature_bc_matrix/",
  model = "max"
)

# H5MU → SingleCellExperiment with altExp("crispr")
sce <- assign_guides(
  counts = "combined.h5mu",
  guide_library = "guides.tsv",
  model = "neg_binomial",
  format = "sce"
)
```

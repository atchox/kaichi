# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# kaichi — perturb-seq analysis tools

A Rust-core library for Perturb-seq analysis with Python and R bindings.
Starting with guide assignment; designed to grow into a broader toolkit.

## Project goals

- Fast, correct guide assignment from 10x Cell Ranger output
- **Single source of truth for both Python and R** — one Rust core, two bindings.
  This is the core mandate: eliminate the drift between Python and R Perturb-seq tooling.
- Ecosystem-compatible output (MuData / Scanpy / scvi-tools in Python; Seurat / Bioconductor in R)
- Designed to scale from single experiments to atlas ingestion

## Positioning

Existing Python Perturb-seq tooling fits per-guide Bayesian mixtures via pyro / SVI,
which carries heavy autodiff overhead for the per-guide problem size. The R side has
no equivalent multi-model toolkit. kaichi reimplements the 11-method catalog with
closed-form EM (not SVI) in Rust with per-guide rayon parallelism. Target: ≥ 20×
speedup over existing pyro-based implementations on common Perturb-seq sizes, with
identical APIs and outputs across Python and R.

Detailed design lives in [docs/design/](docs/design/README.md).

---

## Architecture decisions

### Language / binding stack

```
kaichi-core (Rust, arrow crate)
    ├── kaichi-cli                                  (standalone Rust binary)
    ├── pyo3 + pyo3-arrow + maturin  →  kaichi-py   (Python wheel)
    └── extendr + arrow (R)          →  kaichi-r    (R package)
```

CLI, Python binding, and R binding are siblings above `kaichi-core` — not layered.
The core owns all on-disk I/O so the CLI never needs Python or R at runtime.

- **kaichi-core**: computation, MTX / molecule_info.h5 parsing, EM model fitting,
  H5MU container writing, structural HDF5 copy. Uses `anndata-rs` for AnnData
  read/write of individual modalities.
- **kaichi-cli**: clap-based binary for Snakemake / Nextflow integration.
- **kaichi-py**: pyo3 bindings — produces pip-installable wheels via `maturin develop`.
- **kaichi-r**: extendr bindings — same approach used by rpolars, noodles wrappers.

### In-memory interop format: Apache Arrow

- Use the `arrow` crate in Rust for all internal data representation
- Return `RecordBatch` / `Table` from public Rust API
- Cross the Python boundary via the Arrow C Data Interface (`pyo3-arrow` or `__arrow_c_stream__`)
  — genuinely zero-copy at the binding layer
- R side: more ceremony, may involve a copy; acceptable
- Downstream copies (`.to_pandas()`, HDF5 write) are expected and fine — Arrow zero-copy
  applies at the Rust→Python handoff, not end-to-end

### On-disk format: H5MU primary, H5AD fallback

**H5MU (MuData) is the primary output**, with `mod['rna']` (RNA modality, optionally
passed through from input) and `mod['crispr']` (kaichi's guide assignment). Reasons:

- Perturb-seq is inherently bi-modal — RNA + guides per cell
- `pertpy` and `muon` use this layout natively
- Cleaner than squeezing guide calls into `.obsm` of an H5AD
- Seurat v5 has direct equivalence (Assay objects per modality)

**H5AD is the guide-only fallback** when no RNA companion is provided.

**Three-layer I/O stack** (see [storage-encoding.md](docs/design/storage-encoding.md)):

1. H5MU container: kaichi-owned (small spec, written via `hdf5` crate directly)
2. Per-modality AnnData: delegated to `anndata-rs`
3. Pass-through of unmodified modalities: raw HDF5 `H5Ocopy` (no semantic parsing)

This keeps kaichi-core Python-free for all on-disk I/O.

**Zarr and TileDB-SOMA are out of scope for v0.** Add only when a real user need for
atlas-scale ingestion materializes — and even then, expose as optional Python sinks,
not Rust core deps.

---

## Data model

Authoritative schema lives in [data-model.md](docs/design/data-model.md).
Summary of the per-cell guide call record (Arrow schema):

```
cell_barcode           Utf8
guide_id               Dictionary(Int16, Utf8)    # null if unassigned
target_gene            Dictionary(Int16, Utf8)
umi_count              UInt32
assignment_confidence  Float32                    # model-specific score
assignment_model       Dictionary(Int8, Utf8)
is_unassigned          Boolean
is_multi_infected      Boolean
n_guides_detected      UInt8
```

`is_unassigned` and `is_multi_infected` are mutually exclusive.

---

## Crate selection

| Purpose | Crate |
|---|---|
| Columnar in-memory | `arrow` (Apache Arrow official) |
| AnnData spec read/write | `anndata-rs` (git-pinned; no crates.io release) |
| Raw HDF5 access (H5MU container, H5Ocopy) | `hdf5` or `hdf5-metno` |
| Numerics | `ndarray` |
| Statistical distributions | `statrs` |
| Optimization (Newton-Raphson for NB MLE etc.) | `argmin` |
| Per-guide parallelism | `rayon` |
| CLI | `clap`, `tracing`, `indicatif`, `anyhow` |
| Python bindings | `pyo3` + `pyo3-arrow` + `maturin` |
| R bindings | `extendr-api` + `rextendr` + arrow (R package) |
| DataFrame ergonomics (internal only) | `polars` (optional, uses Arrow natively) |

---

## What to avoid

- Writing a custom binary format
- Returning JSON or CSV from Rust bindings (kills zero-copy advantage)
- Loom format (older, less ecosystem support)
- Making TileDB-SOMA or Zarr a hard dependency of the Rust core
- Claiming end-to-end zero-copy (copies happen at pandas conversion, HDF5 write, etc.)

---

## Format landscape context (for future reference)

| Format | Best for |
|---|---|
| H5AD | Standard local analysis, sharing, <1M cells |
| Zarr | Cloud storage (S3/GCS), parallel access, atlas-scale |
| TileDB-SOMA | Multi-experiment atlas queries, CELLxGENE Census-style |
| Arrow/Parquet | In-memory interop, intermediate checkpoints |
| Lance | Single-cell AI/ML training workloads (emerging) |

TileDB-SOMA becomes relevant when aggregating guide calls + gene expression across many
experiments into a queryable atlas. At that point, expose it as an optional sink.

---

## Computational bottlenecks (actual, post-Cell-Ranger)

kaichi consumes Cell Ranger output — it does NOT do FASTQ/BAM parsing. The work
that matters is:

1. **Per-guide mixture model fitting** (closed-form EM, not SVI).
   This is the dominant cost and the reason for Rust + rayon.
2. **molecule_info.h5 UMI dedup** (optional path) — Rust reads the HDF5 directly.
3. **Sparse matrix construction** from Cell Ranger MTX or `.X` — cheap, but worth
   keeping zero-copy where possible.
4. **Arrow handoff to bindings** — not actually a bottleneck (data sizes are small)
   but matters for "single source of truth" guarantee.
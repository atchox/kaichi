# kaichi — Design Overview

## What kaichi does

kaichi assigns CRISPR guide RNAs to cells from 10x Genomics Perturb-seq data. It takes
Cell Ranger feature-barcode output and a guide library reference, runs one of several
assignment models, and writes a MuData file (Python) or Seurat object (R) with guide
calls attached.

It does **not** handle RNA expression — that is Cell Ranger's job. kaichi picks up where
Cell Ranger leaves off.

## Why kaichi exists

Existing Python Perturb-seq tooling fits per-guide Bayesian mixtures via pyro / SVI,
which carries heavy autodiff overhead for the per-guide problem size (a few thousand
cells, a small 2-component mixture). The R side has no equivalent multi-model toolkit;
Seurat's mixscape and the Bioconductor ecosystem don't share implementations with
Python tools, so users on different sides see different answers from nominally the
same method.

kaichi's positioning:

1. **Fast** — closed-form EM (not SVI) in Rust, with per-guide parallelism via rayon.
   Target ≥ 20× speedup over existing pyro-based implementations on common Perturb-seq
   sizes.
2. **Single source of truth across Python and R.** One Rust implementation, two
   bindings. Eliminates the silent drift that comes from re-implementing models per
   language.
3. **Multi-modal-native output.** MuData (Python) and Seurat v5 (R) — both have
   first-class slots for a CRISPR modality.

## Non-goals (v1)

- RNA / gene expression quantification
- Differential expression or downstream statistical testing
- FASTQ alignment or Cell Ranger replacement
- Direct-capture (non-10x) guide sequencing
- GPU acceleration (CPU + rayon should suffice for the per-guide workload)

## Architecture

```
Cell Ranger output (MTX / molecule_info.h5)
Guide library TSV
        │
        ▼
┌─────────────────────────────────────────┐
│        kaichi-core (Rust)               │
│  - count matrix parsing (no Python dep) │
│  - UMI deduplication (molecule_info)    │
│  - assignment model execution           │
│  - per-guide rayon parallelism          │
│  - H5AD I/O via anndata-rs              │
│  - H5MU container written directly      │
│  - structural pass-through via H5Ocopy  │
│  - returns Arrow RecordBatch to bindings│
└────────────────┬────────────────────────┘
                 │ Arrow C Data Interface (zero-copy)
        ┌────────┼────────┬────────────────┐
        ▼        ▼        ▼                ▼
   kaichi-cli  kaichi-py (pyo3)   kaichi-r (extendr)
   (standalone)
                MuData / H5MU      Seurat (default)
                AnnData / H5AD     SingleCellExperiment (option)
```

The CLI, Python binding, and R binding are siblings above `kaichi-core` — not
layered. The core owns all on-disk I/O so the CLI never needs Python or R.

## Repo layout

```
kaichi/
├── kaichi-core/        Rust library — models, Arrow plumbing, HDF5 I/O
├── kaichi-cli/         standalone Rust binary
├── kaichi-py/          pyo3 bindings → `kaichi` Python wheel
├── kaichi-r/           extendr bindings → `kaichi` R package
├── docs/design/        these documents
└── tests/              shared fixtures + cross-language equivalence tests
```

Performance benchmarking is run externally (omnibenchmark), not from this repo.

Cargo workspace, one Git repo. Releases produce Python wheels (maturin) and an R
source package (rextendr) from the same `kaichi-core` version.

## Technology choices

| Layer | Choice | Reason |
|---|---|---|
| Core | Rust | EM speed + per-guide rayon; clean dual-binding story |
| Interop | Apache Arrow (`arrow` crate) | Zero-copy handoff to Python/R |
| Numerics | `ndarray`, `statrs`, `argmin` | Mature, no autodiff overhead |
| Parallelism | `rayon` | Embarrassingly parallel across guides |
| HDF5 raw access | `hdf5` crate | For H5Ocopy structural pass-through |
| AnnData I/O | `anndata-rs` (git-pinned, vendored if needed) | Spec-correct AnnData reading/writing |
| H5MU container | written directly via `hdf5` | anndata-rs does not support MuData |
| CLI | `clap` + `tracing` + `indicatif` | Standalone Rust binary |
| Python binding | `pyo3` + `pyo3-arrow` + `maturin` | Standard, pip-wheel compatible |
| R binding | `extendr` + arrow R package | Same path as rpolars |
| Python output | MuData (H5MU) primary, AnnData (H5AD) fallback | `pertpy`/`muon` native |
| R output | Seurat v5 primary, SingleCellExperiment optional | mixscape audience |

## Cross-references

- Input/output formats, API sketches → [io-spec.md](io-spec.md)
- Arrow schema, MuData / Seurat structure → [data-model.md](data-model.md)
- Model catalog → [assignment-models.md](assignment-models.md)
- HDF5 read/write, anndata-rs use, structural copy → [storage-encoding.md](storage-encoding.md)
- Arrow handoff between Rust and Python/R → [binding-interop.md](binding-interop.md)
- Standalone CLI design → [cli.md](cli.md)
- Validation against baselines, ship criteria → [validation.md](validation.md)

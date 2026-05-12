# Test fixtures

This directory is gitignored. Its contents are populated automatically when
integration tests run via `cargo test`.

## What gets downloaded

`crispat/` — extracted from a tarball of `velten-group/crispat` main branch:
- `example_data/Schraivogel/gRNA_counts.h5ad` — guide count matrix input
- `example_data/guide_assignments/` — per-model reference assignments

The download runs once per test process; a `.ready` marker prevents re-downloading
on subsequent runs.

## Parameter mapping

| kaichi model   | Reference dir        | Reference file         | Parameter match              |
|----------------|----------------------|------------------------|------------------------------|
| `umi`          | `UMI/`               | `assignments_t5.csv`   | `umi_threshold=5` (default)  |
| `max`          | `maximum/`           | `assignments.csv`      | default params               |
| `ratio`        | `ratios/`            | `assignments_t0.3.csv` | `min_fraction=0.3` (default) |
| `neg_binomial` | `negative_binomial/` | `assignments.csv`      | default params               |
| `poisson_gauss`| `poisson_gauss/`     | `assignments.csv`      | default params               |

## Column layout of reference CSVs

Most models: `cell, gRNA, UMI_counts`  
Ratio model: `cell, percent_counts, gRNA, UMI_counts`

`load_reference` locates the gRNA column by header name to handle this difference.

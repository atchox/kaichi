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

| kaichi model    | Reference dir        | Reference file          | Parameter match               |
|-----------------|----------------------|-------------------------|-------------------------------|
| `umi`           | `UMI/`               | `assignments_t5.csv`    | `umi_threshold=5` (default)   |
| `max`           | `maximum/`           | `assignments.csv`       | default params                |
| `ratio`         | `ratios/`            | `assignments_t0.3.csv`  | `min_fraction=0.3` (default)  |
| `poisson_gauss` | `poisson_gauss/`     | `assignments.csv`       | default params                |
| `poisson`       | `poisson/`           | `assignments.csv`       | default params                |
| `neg_binomial`  | `negative_binomial/` | `assignments.csv`       | default params                |
| `binomial`      | `binomial/`          | `assignments.csv`       | default params                |
| `beta2`         | `2-BetaMM/`          | `assignments.csv`       | default params                |
| `beta3`         | `3-BetaMM/`          | `assignments.csv`       | default params                |
| `quantiles`     | `quantiles/`         | `assignments_t0.1.csv`  | `quantile=0.1` (not default)  |

Notes:
- The `gauss/` fixture directory has no kaichi equivalent; crispat's `ga_gauss` is a
  pure Gaussian mixture not exposed as a standalone model in kaichi.
- The kaichi `quantiles` default is `quantile=0.05` but no `t0.05` reference exists.
  Equivalence tests use `quantile=0.1` to match `assignments_t0.1.csv`.

## Column layout of reference CSVs

The gRNA column name varies across models. `load_reference` locates it by header name:

| Reference dir    | gRNA column name |
|------------------|-----------------|
| `UMI/`           | `gRNA`          |
| `maximum/`       | `gRNA`          |
| `ratios/`        | `gRNA`          |
| `poisson_gauss/` | `gRNA`          |
| `poisson/`       | `gRNA`          |
| `negative_binomial/` | `gRNA`      |
| `binomial/`      | `gRNA`          |
| `2-BetaMM/`      | `gRNA`          |
| `3-BetaMM/`      | `gRNA`          |
| `quantiles/`     | `gRNA`          |

The `ratios/` files include an extra `percent_counts` column before `gRNA`.
The `binomial/` and `poisson/` files include model internals (probabilities, means)
before `gRNA`; the column is still located by name.

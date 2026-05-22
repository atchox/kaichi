# kaichi — Python

CRISPR guide assignment for Perturb-seq, as a Python library.

## Install

```bash
pip install kaichi
```

Wheels are available for Linux x86_64, macOS arm64, and macOS x86_64 (Python ≥ 3.11).

## Quick start

```python
import kaichi

result = kaichi.assign("gRNA_counts.h5ad", model="poisson_gauss")
print(result)
# pyarrow Table: cell_barcode, guide_id, umi_count,
#                assignment_confidence, is_unassigned,
#                is_multi_infected, n_guides_detected
```

Convert to pandas or polars:

```python
df = result.to_pandas()
```

## API

### `kaichi.assign(path, model="poisson_gauss", n_jobs=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Path to an `.h5ad` guide-count file |
| `model` | `str` | `"poisson_gauss"` | Assignment model (see table below) |
| `n_jobs` | `int \| None` | `None` | Worker threads; `None` = half of logical cores |

Returns a `pyarrow.Table` with one row per cell.

### Output columns

| Column | Type | Notes |
|---|---|---|
| `cell_barcode` | string | |
| `guide_id` | string | null if unassigned |
| `umi_count` | uint32 | null if unassigned |
| `assignment_confidence` | float32 | posterior probability or proportion; null if unassigned |
| `is_unassigned` | bool | |
| `is_multi_infected` | bool | cell passes threshold for more than one guide |
| `n_guides_detected` | uint8 | guides above threshold, regardless of final assignment |

`is_unassigned` and `is_multi_infected` are mutually exclusive.

## Models

| Model | Type | When to use |
|---|---|---|
| `umi` | Threshold | Fast baseline; assign any guide ≥ N UMIs |
| `max` | Deterministic | Assign the single highest-count guide; ties → unassigned |
| `ratio` | Threshold | Assign if top guide UMIs / total UMIs > fraction |
| `poisson_gauss` | EM mixture | Good default; Poisson background, log-normal signal |
| `poisson` | EM mixture | Depth-normalised Poisson mixture |
| `neg_binomial` | EM mixture | Like `poisson` but handles overdispersed counts; recommended for noisy libraries |
| `binomial` | EM mixture | Models guide fraction (count / total guide UMIs) |
| `beta2` | EM mixture | 2-component Beta mixture on per-cell max guide proportion |
| `beta3` | EM mixture | 3-component Beta mixture; separates low / intermediate / high |
| `quantiles` | Rank-based | Assign top Q% of cells per guide by proportion |

Mixture models fit one model per guide in parallel and assign cells where the
posterior probability of the signal component exceeds `min_confidence` (default 0.8
for count-based models, 0.5 for Beta models).

## Input format

An `.h5ad` file with:

- `obs_names` — cell barcodes
- `var_names` — guide IDs
- `X` — sparse count matrix (cells × guides)

This is the `crispr_gene_expression` feature-barcode matrix produced by Cell Ranger.

## Version

```python
import kaichi
print(kaichi.__version__)
```

# kaichi — Python

[![PyPI](https://img.shields.io/pypi/v/kaichi)](https://pypi.org/project/kaichi/)
[![Python](https://img.shields.io/pypi/pyversions/kaichi)](https://pypi.org/project/kaichi/)
[![License](https://img.shields.io/github/license/atchox/kaichi)](https://github.com/atchox/kaichi/blob/main/LICENSE)

CRISPR guide assignment for Perturb-seq, as a Python library.

## Install

```bash
pip install kaichi
```

Wheels are available for Linux x86_64, macOS arm64, and macOS x86_64 (Python ≥ 3.11).

## Quick start

```python
import kaichi

# One-shot assign — returns an AnnData
adata = kaichi.assign("gRNA_counts.h5ad", model="poisson_gauss")

# Two-stage: fit once, threshold multiple times
scores = kaichi.score("gRNA_counts.h5ad", model="poisson_gauss")
strict = kaichi.decide(scores, min_confidence=0.9)   # pyarrow.RecordBatch
lenient = kaichi.decide(scores, min_confidence=0.7)  # reuses cached scores
```

## API

### `kaichi.assign(path, model="poisson_gauss", *, min_confidence=None, quantile=None, n_jobs=None)`

Fit and assign in one step. Returns an `anndata.AnnData` with:

- `.X` — raw UMI counts (sparse CSR, uint32)
- `.layers["assigned"]` — binary sparse CSR, 1 where assigned
- `.obs` — per-cell assignment columns (see table below)
- `.uns["kaichi"]` — provenance (`model`, `model_params`, `version`)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Path to an `.h5ad` guide-count file |
| `model` | `str` | `"poisson_gauss"` | Assignment model (see table below) |
| `min_confidence` | `float \| None` | `None` | Override posterior threshold |
| `quantile` | `float \| None` | `None` | Top-fraction threshold for the `quantiles` model |
| `n_jobs` | `int \| None` | `None` | Worker threads; `None` = half of logical cores |

### `kaichi.score(path, model="poisson_gauss", *, n_jobs=None)`

Run the EM fitting stage only; return a `kaichi.ScoreResult`. Raises `ValueError` for
single-stage models (`umi`, `ratio`, `max`) — use `assign()` for those.

### `kaichi.decide(scores, min_confidence=0.9)`

Apply a confidence threshold to a cached `ScoreResult`. Returns a
`pyarrow.RecordBatch` with one row per cell. Call multiple times with different
thresholds without re-fitting.

### `.obs` / output columns

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

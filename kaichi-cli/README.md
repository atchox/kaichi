# kaichi — CLI

[![License](https://img.shields.io/github/license/atchox/kaichi)](https://github.com/atchox/kaichi/blob/main/LICENSE)

CRISPR guide assignment for Perturb-seq, as a command-line tool.

## Install

```bash
cargo install kaichi-cli
```

Or build from source:

```bash
cargo build --release -p kaichi-cli
# binary at target/release/kaichi
```

## Quick start

One-shot (fit + threshold + write):

```bash
kaichi assign --counts gRNA_counts.h5ad --output assignments.h5ad
```

Split flow — fit once, threshold many times without re-running EM:

```bash
# Fit and cache the score matrix to disk.
kaichi score --counts gRNA_counts.h5ad --model poisson_gauss --output scored.h5ad

# Threshold the cache — runs in milliseconds.
kaichi decide --scores scored.h5ad --min-confidence 0.9 --output strict.h5ad
kaichi decide --scores scored.h5ad --min-confidence 0.7 --output lenient.h5ad
```

The cache (`scored.h5ad`) is a regular AnnData file: `X` holds the preserved
UMI counts, `layers/scores` holds the float32 posteriors, and `uns/kaichi`
records the model name and fitted parameters. The `decide` step pulls the
model identity from `uns/kaichi` automatically — you don't repeat `--model`.

To cap worker threads on shared HPC nodes, add `--threads N` to any subcommand.

## Subcommands

### `kaichi assign`

Fit a model and write the final assignment in one step.

| Flag | Default | Description |
|---|---|---|
| `--counts <PATH>` | — | Input `.h5ad` guide-count file (required) |
| `--output <PATH>` | — | Output path: `.h5ad` or `.csv` |
| `--model <NAME>` | `poisson_gauss` | Assignment model (see table below) |
| `--threads <N>` | half of logical cores | Rayon worker thread cap |

### `kaichi score`

Fit a two-stage model and cache the score matrix. Single-stage models
(`umi`, `ratio`, `max`) are rejected — use `assign` for those.

| Flag | Default | Description |
|---|---|---|
| `--counts <PATH>` | — | Input `.h5ad` guide-count file (required) |
| `--model <NAME>` | `poisson_gauss` | One of `poisson_gauss`, `poisson`, `neg_binomial`, `binomial` |
| `--output <PATH>` | — | Scored `.h5ad` output (required) |
| `--threads <N>` | half of logical cores | Rayon worker thread cap |

### `kaichi decide`

Threshold a cached score H5AD into final assignments. The output extends the
input with a `layers/assigned` group plus assignment obs columns.

| Flag | Default | Description |
|---|---|---|
| `--scores <PATH>` | — | Scored `.h5ad` from `kaichi score` (required) |
| `--min-confidence <F>` | — | Posterior threshold in [0, 1] (required) |
| `--output <PATH>` | — | Output `.h5ad` (mutually exclusive with `--in-place`) |
| `--in-place` | off | Overwrite the input scored H5AD with the decided result |
| `--threads <N>` | half of logical cores | Rayon worker thread cap |

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

## Output

`assign` and `decide` produce the same shape, one row per cell:

| Field | Type | Notes |
|---|---|---|
| `cell_barcode` | string | |
| `guide_id` | string | null if unassigned |
| `umi_count` | uint32 | null if unassigned |
| `assignment_confidence` | float32 | posterior probability or proportion; null if unassigned |
| `is_unassigned` | bool | |
| `is_multi_infected` | bool | cell passes threshold for more than one guide |
| `n_guides_detected` | uint8 | guides above threshold, regardless of final assignment |

`is_unassigned` and `is_multi_infected` are mutually exclusive.

`.h5ad` output writes the full assignment table into `obs`. `.csv` output writes
`cell_barcode`, `guide_id`, and `umi_count` only.

`score` writes a partial H5AD with `X` = preserved UMI counts,
`layers/scores` = float32 posteriors, and `uns/kaichi` carrying the model
name and fitted parameters. `decide` reads this back and adds
`layers/assigned`, the obs columns above, and `uns/kaichi/min_confidence`.

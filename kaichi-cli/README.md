# kaichi — CLI

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

```bash
kaichi assign \
  --counts gRNA_counts.h5ad \
  --output assignments.h5ad
```

The default model is `poisson_gauss`. To use a different one:

```bash
kaichi assign \
  --counts gRNA_counts.h5ad \
  --output assignments.h5ad \
  --model neg_binomial
```

Cap the number of worker threads (useful on shared HPC nodes):

```bash
kaichi assign \
  --counts gRNA_counts.h5ad \
  --output assignments.h5ad \
  --threads 4
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--counts <PATH>` | — | Input `.h5ad` guide-count file (required) |
| `--output <PATH>` | — | Output path: `.h5ad` or `.csv` (required) |
| `--model <NAME>` | `poisson_gauss` | Assignment model (see table below) |
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

Each row is one cell. Key fields:

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

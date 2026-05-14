# kaichi

CRISPR guide assignment for Perturb-seq. Takes a guide-count matrix from Cell Ranger
and produces a per-cell assignment table — which guide each cell received, with
confidence scores and multi-infection flags.

## Quick start

```bash
cargo build --release -p kaichi-cli

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

Output can be `.h5ad` (full AnnData with all fields) or `.csv` (barcode, guide, UMI count).

## Input

A guide-count AnnData file (`.h5ad`) with:

- `obs_names` — cell barcodes
- `var_names` — guide IDs
- `X` — sparse count matrix (cells × guides)

This is the `crispr_gene_expression` feature-barcode matrix produced by Cell Ranger.

## Models

| Model | Type | When to use |
|---|---|---|
| `umi` | Threshold | Fast baseline; assign any guide ≥ N UMIs |
| `max` | Deterministic | Assign the single highest-count guide; ties → unassigned |
| `ratio` | Threshold | Assign if top guide UMIs / total guide UMIs > fraction |
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

## Build and test

```bash
cargo build
cargo test
```

Run a single model's tests:

```bash
cargo test -p kaichi-core neg_binomial
```

Run fixture-backed equivalence tests against crispat reference outputs:

```bash
cargo test -p kaichi-core --test equivalence
```

Fixtures are downloaded into `tests/fixtures/crispat/` on first run (git-ignored).
See [tests/FIXTURES.md](tests/FIXTURES.md) for the reference files and
parameter mapping.

## Design documents

[docs/design/](docs/design/README.md) covers architecture, the full model catalog
with algorithm details, I/O spec, and the Python/R binding design.

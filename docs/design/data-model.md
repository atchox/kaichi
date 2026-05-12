# kaichi — Data Model

## Arrow schema: guide call record

This is the schema of the Arrow `RecordBatch` returned by the Rust core.
The Python and R bindings construct their native objects from this batch.

```
cell_barcode          Utf8
guide_id              Dictionary(Int16, Utf8)    # categorical — most cells share a small set
target_gene           Dictionary(Int16, Utf8)    # categorical
umi_count             UInt32                     # UMIs for the assigned guide
assignment_confidence Float32                    # model-specific score in [0, 1]
assignment_model      Dictionary(Int8, Utf8)     # which model produced this call
is_unassigned         Boolean                    # true if no guide meets assignment criteria
is_multi_infected     Boolean                    # true if multiple guides detected above threshold
n_guides_detected     UInt8                      # number of guides with UMI count > background
```

`guide_id` is null when `is_unassigned = true`.
`target_gene` is null when `is_unassigned = true` or when no guide-library metadata
was supplied for the assigned guide.
`is_multi_infected` and `is_unassigned` are mutually exclusive.

## MuData structure: `mod['crispr']` AnnData

```
.X          sparse UInt32             cells × guides, raw UMI counts
.obs        DataFrame                 one row per cell (columns = Arrow schema above)
.var        DataFrame                 one row per guide (guide IDs plus optional metadata)
.uns['kaichi']  dict                  run provenance (see below)
```

### `.var` columns

Minimal guide-ID-only output:

```
guide_id              index (Utf8)
```

When a guide library is supplied, kaichi adds metadata columns:

```
target_gene           Utf8
sequence              Utf8
chromosome            Utf8        (optional, null if not in reference)
cut_site              Int64       (optional)
strand                Utf8        (optional, "+" / "-")
on_target_score       Float32     (optional)
is_non_targeting      Boolean     (derived: target_gene in {"non-targeting", "NTC", "safe-harbor"})
```

`target_gene`, `sequence`, and derived metadata are nullable or absent when no guide
library is provided. The assignment itself remains valid because `guide_id` comes
from the count matrix.

### `.uns['kaichi']`

```json
{
  "version": "0.1.0",
  "model": "neg_binomial",
  "model_params": { "min_confidence": 0.8, "max_em_iters": 100, "init_seed": 2024 },
  "input_source": "filtered_feature_bc_matrix",
  "n_cells_input": 10842,
  "n_cells_assigned": 9201,
  "n_cells_unassigned": 412,
  "n_cells_multi_infected": 1229,
  "timestamp": "2026-05-12T00:00:00Z"
}
```

## Cell barcode handling

Barcodes are stored as plain strings (e.g., `ACGTACGTACGT-1`).
10x suffix (the `-1` gem group) is preserved as-is — kaichi does not strip or modify it.
Callers are responsible for matching barcodes between modalities if suffixes differ.

## Multi-infection representation

Multi-infected cells (`is_multi_infected = true`) have a single row in `.obs`.
`guide_id` holds the guide with the highest UMI count; `n_guides_detected` records how many
guides were above the detection threshold.

The full per-guide UMI counts for a cell are always recoverable from `.X`.
Exploding multi-infected cells to multiple rows is a downstream operation, not done by kaichi.

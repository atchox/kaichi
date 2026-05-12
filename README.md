# kaichi

Rust workspace for Perturb-seq CRISPR guide assignment.

The implemented core operates on Arrow `RecordBatch` values, reads guide-count H5AD
input, and currently provides four assignment models: `umi`, `max`, `ratio`, and
`poisson_gauss`.
The repository also contains CLI, Python, and R crates that share the same workspace
version and depend on `kaichi-core`.

## Build and Test

```bash
cargo build
cargo test
```

Run only the core crate:

```bash
cargo test -p kaichi-core
```

Run the CLI help:

```bash
cargo run -p kaichi-cli -- --help
```

## Crates

```text
kaichi-core   Assignment schemas, H5AD input, model implementations
kaichi-cli    `kaichi` binary command surface
kaichi-py     Python extension crate
kaichi-r      R extension crate
```

`kaichi-core` is the source of truth for model behavior. Bindings and the CLI sit
above the core rather than reimplementing assignment logic.

## Implemented Core

The core model trait is defined in
[kaichi-core/src/models/mod.rs](kaichi-core/src/models/mod.rs):

```rust
pub trait AssignmentModel: Send + Sync {
    fn name(&self) -> &'static str;
    fn assign(&self, input: &AssignmentInput) -> anyhow::Result<RecordBatch>;
    fn params_json(&self) -> serde_json::Value;
}
```

`AssignmentInput` contains:

```text
counts      RecordBatch(cell_barcode: Utf8, guide_id: Utf8, umi_count: UInt32)
covariates  RecordBatch(cell_barcode: Utf8, batch: Utf8, total_counts: Float32)
```

Guide metadata is not required for assignment. Counts provide the guide IDs and UMI
values needed by the model. The intended enrichment step accepts an optional guide
library to validate guide IDs and populate fields such as `target_gene` and
`sequence`; guide-ID-only outputs remain valid.

Implemented models:

| Model | File | Parameters | Behavior |
|---|---|---|---|
| `umi` | [umi.rs](kaichi-core/src/models/umi.rs) | `umi_threshold`, default `5` | Assign guides with count >= threshold; multi-hit cells are marked multi-infected |
| `max` | [max.rs](kaichi-core/src/models/max.rs) | `umi_threshold`, default `0` | Assign the unique highest-count guide; ties are unassigned |
| `ratio` | [ratio.rs](kaichi-core/src/models/ratio.rs) | `min_fraction`, default `0.3` | Assign the top guide if `top_umi / total_umi > min_fraction` |
| `poisson_gauss` | [poisson_gauss.rs](kaichi-core/src/models/poisson_gauss.rs) | `min_confidence`, default `0.5`; EM controls | Per-guide Poisson background / Gaussian log-count signal mixture |

## Output Schema

All models return one Arrow `RecordBatch` using the schema in
[kaichi-core/src/schema.rs](kaichi-core/src/schema.rs):

```text
cell_barcode           Utf8
guide_id               Dictionary(Int16, Utf8)
target_gene            Dictionary(Int16, Utf8)
umi_count              UInt32
assignment_confidence  Float32
assignment_model       Dictionary(Int8, Utf8)
is_unassigned          Boolean
is_multi_infected      Boolean
n_guides_detected      UInt8
```

`guide_id`, `target_gene`, `umi_count`, and `assignment_confidence` are nullable for
unassigned cells. `target_gene` is also nullable when no guide metadata was supplied.

## H5AD Input

[kaichi-core/src/io/h5ad.rs](kaichi-core/src/io/h5ad.rs) reads guide-count AnnData
files through `anndata-rs`.

Expected input:

```text
obs_names   cell barcodes
var_names   guide IDs
X           sparse CSR guide-count matrix, cells x guides
```

The reader expands nonzero entries into the long-form Arrow `counts` batch and
computes `total_counts` per cell for the covariate batch. `batch` is currently set to
the default category.

## Tests

Run all tests:

```bash
cargo test
```

Run only core tests:

```bash
cargo test -p kaichi-core
```

Run only one model's unit tests:

```bash
cargo test -p kaichi-core umi
cargo test -p kaichi-core max
cargo test -p kaichi-core ratio
cargo test -p kaichi-core poisson_gauss
```

Run fixture-backed equivalence tests:

```bash
cargo test -p kaichi-core --test equivalence
```

The equivalence tests download the crispat Schraivogel example fixture into
`tests/fixtures/crispat/` on first run. The fixture directory is ignored by Git.
See [tests/FIXTURES.md](tests/FIXTURES.md) for the reference files and parameter
mapping.

## CLI Surface

The CLI crate defines the intended command surface:

```bash
cargo run -p kaichi-cli -- --help
```

Available commands:

```text
kaichi assign
kaichi validate
```

The intended `assign` contract is counts-first:

```bash
kaichi assign \
  --counts gRNA_counts.h5ad \
  --output assignments.h5ad \
  --model umi
```

Guide metadata is optional:

```bash
kaichi assign \
  --counts gRNA_counts.h5ad \
  --guide-library guides.tsv \
  --output assignments.h5ad \
  --model umi
```

At this stage the commands parse arguments and report that execution is not yet
implemented.

## Design Documents

Detailed design notes are in [docs/design](docs/design/README.md):

```text
overview.md           architecture and positioning
io-spec.md            inputs, outputs, and API sketches
data-model.md         Arrow schema and object layout
assignment-models.md  model catalog and algorithms
storage-encoding.md   H5AD/H5MU storage design
binding-interop.md    Arrow boundary for Python/R
cli.md                command-line interface design
validation.md         equivalence and ship criteria
```

# kaichi — Design Documents

Read in order:

1. [overview.md](overview.md) — what kaichi does, why it exists, repo layout, tech stack
2. [io-spec.md](io-spec.md) — inputs, outputs, Python and R API sketches
3. [data-model.md](data-model.md) — Arrow schema, MuData / Seurat structure, provenance
4. [assignment-models.md](assignment-models.md) — the 11-method catalog and the EM-not-SVI plan
5. [storage-encoding.md](storage-encoding.md) — HDF5 read/write, anndata-rs role, structural copy
6. [binding-interop.md](binding-interop.md) — Arrow C Data Interface handoff, sparse matrix encoding
7. [cli.md](cli.md) — standalone Rust binary, Snakemake/Nextflow integration
8. [validation.md](validation.md) — equivalence with baseline implementations, ship criteria (performance benchmarking is external, via omnibenchmark)

These are living design documents. Update them when decisions change rather than
letting code and docs drift.

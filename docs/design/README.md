# kaichi — Design Documents

Read in order:

1. [overview.md](overview.md) — what kaichi does, why it exists, repo layout, tech stack
2. [io-spec.md](io-spec.md) — inputs, outputs, Python and R API sketches
3. [data-model.md](data-model.md) — Arrow schema, MuData / Seurat structure, provenance
4. [assignment-models.md](assignment-models.md) — the 11-method catalog and the EM-not-SVI plan
5. [storage-encoding.md](storage-encoding.md) — HDF5 read/write, AnnData spec implementation, structural copy
6. [memory-layout.md](memory-layout.md) — in-memory data flow: one buffer per piece of data, typed views over it
7. [binding-interop.md](binding-interop.md) — Arrow C Data Interface handoff, sparse matrix encoding
8. [cli.md](cli.md) — standalone Rust binary, Snakemake/Nextflow integration
9. [validation.md](validation.md) — equivalence with baseline implementations, ship criteria (performance benchmarking is external, via omnibenchmark)
10. [score-decide.md](score-decide.md) — v0.2: split score/decide, Arrow-native `ScoreResult`, EM mixture refactor

These are living design documents. Update them when decisions change rather than
letting code and docs drift.

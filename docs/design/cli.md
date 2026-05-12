# kaichi — CLI

A standalone Rust binary, `kaichi`. No Python or R runtime required at any point.
Designed to plug into Snakemake / Nextflow / shell-script pipelines.

---

## Why a CLI

Most Perturb-seq pipelines are orchestrated by Snakemake or Nextflow, where each
step is a process call. Forcing those pipelines to depend on a Python or R
runtime *just to invoke guide assignment* adds friction:

- Environment management (conda envs per step)
- Slow startup time (Python import overhead)
- Coupling between pipeline and analysis language

A self-contained Rust binary is the canonical answer — `samtools`, `bcftools`,
`bwa`, `bedtools`, `nf-core/cellranger-wrappers` all follow this pattern.

---

## Scope

The CLI is a **thin wrapper around the Rust core**. Everything the library can
do, the CLI can do. The bindings (`kaichi-py`, `kaichi-r`) and the CLI all sit
above the same Rust core; they are siblings, not layered.

```
                ┌────────────────────┐
                │   kaichi-core      │
                └─────────┬──────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   kaichi-cli       kaichi-py          kaichi-r
   (Rust binary)    (Python wheel)     (R package)
```

The CLI does no file-format conversion that the library doesn't already do.
On-disk read/write is fully owned by `kaichi-core` (see
[storage-encoding.md](storage-encoding.md)).

---

## Crate setup

```
kaichi-cli/
├── Cargo.toml          (binary crate)
└── src/
    └── main.rs         (clap parsing → dispatch to kaichi-core functions)
```

Library deps:
- `clap` (derive feature) — argument parsing
- `tracing` + `tracing-subscriber` — structured logging
- `indicatif` — progress bars for the EM fitting loop
- `anyhow` — error reporting at the binary boundary

---

## Command surface

```
kaichi --help
kaichi --version

kaichi assign \
    --counts <PATH>            # Cell Ranger dir, molecule_info.h5, h5ad, or h5mu
    --output <PATH>            # .h5mu, .h5ad
    --model <NAME>             # umi, max, ratio, neg_binomial, poisson_gauss, ...
    [--guide-library <TSV>]    # optional guide metadata / validation
    [--input <PATH>]           # existing h5ad / h5mu to attach the crispr modality to
    [--params <JSON or @file>] # model-specific overrides
    [--barcode-join inner|left|outer]  # default: inner
    [--threads N]              # default: all available
    [--seed N]                 # default: 2024
    [--verbose / -v]
    [--quiet  / -q]

kaichi validate \
    --a <PATH>                 # assignments to compare (H5MU, H5AD, or CSV)
    --b <PATH>                 # assignments to compare against (H5MU, H5AD, or CSV)
    [--tolerance 0.99]         # min confident-call agreement fraction
    [--report <PATH>]          # write a per-guide breakdown CSV
```

`assign` is the workhorse. `validate` is for users who want to compare two
assignment outputs — typical uses:

- regression check after upgrading kaichi ("did my v0.2 → v0.3 upgrade change calls?")
- cross-tool sanity check against a published assignments table
- comparing two models on the same input (`kaichi assign --model A` vs `--model B`)

Both `--a` and `--b` accept any of: H5MU (reads `mod['crispr'].obs`), H5AD
(reads `.obs`), or a simple CSV with columns `(cell, gRNA)` for ingesting
non-kaichi outputs. Exit code 0 if confident-call agreement ≥ tolerance,
non-zero otherwise.

Performance benchmarking is **out of scope for kaichi itself** — it's run
externally via omnibenchmark, which only needs `kaichi assign` to behave well
on stdin/stdout/exit codes.

`--counts` is the only biological input required for assignment. `--guide-library`
is optional: when supplied, kaichi validates counted guide IDs against it and carries
metadata such as `target_gene` and `sequence` into the output. Without it, kaichi
emits guide-ID-only assignments with nullable or absent annotation fields.

---

## Parameter overrides

`--params` accepts a JSON object literal or `@path.json`:

```
kaichi assign \
    ... \
    --model neg_binomial \
    --params '{"min_confidence": 0.9, "max_em_iters": 200}'

kaichi assign \
    ... \
    --model neg_binomial \
    --params @nb_params.json
```

Keys map to the parameter names documented per model in
[assignment-models.md](assignment-models.md). Unknown keys fail loud at parse time.

---

## Output

`--output out.h5mu` writes a MuData file following the structure in
[data-model.md](data-model.md). `--output out.h5ad` writes a single-modality
AnnData (guide-only).

If `--guide-library` is omitted, `.var` is built from guide IDs observed in the count
input. If it is provided, `.var` is enriched with the TSV metadata.

If `--input` is provided, the input file's RNA modality is included in the
output via structural HDF5 copy (see [storage-encoding.md](storage-encoding.md)).

If `--input` is omitted and `--output` is `.h5mu`, the file contains only
`mod['crispr']`. That's valid MuData, just minimal.

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic error (file not found, parse failure, invalid args) |
| 2 | Input validation failed (guide-library mismatch if supplied, no shared barcodes) |
| 3 | Model fitting failed (EM divergence on all guides, etc.) |
| 4 | I/O error during read / write |
| 64 | Usage error (clap's default for misuse) |

Standard Unix convention: non-zero means failure; the precise code lets
Snakemake `onerror` blocks distinguish failure modes without parsing logs.

---

## Logging

Use `tracing` with three default levels:

- **default** (no flag): one summary line per major phase (load, fit, write),
  plus warnings and errors.
- `-v` / `--verbose`: per-guide progress, per-phase timings.
- `-vv`: debug-level — per-EM-iteration log-likelihood, per-cell counts.
- `-q` / `--quiet`: only errors.

Format respects the terminal: `tracing-subscriber` with `fmt::layer()` for
TTY, plain JSON Lines if `KAICHI_LOG_JSON=1` or output is not a TTY. The JSON
form makes Snakemake log parsing trivial.

Progress bars (`indicatif`) shown by default during the per-guide EM loop;
suppressed automatically when stdout is not a TTY or `-q` is set.

---

## Threading

`--threads N` controls `rayon`'s thread pool. Default: all available cores.
Set `RAYON_NUM_THREADS` env var as fallback.

The per-guide work is embarrassingly parallel, so wall-clock scales near-linearly
with cores until guide count saturates available threads.

---

## Snakemake integration sketch

```python
rule kaichi_assign:
    input:
        counts = "data/cellranger/{sample}/filtered_feature_bc_matrix/",
        rna = "data/rna/{sample}.h5ad",
        guides = "config/guide_library.tsv",
    output:
        h5mu = "results/{sample}.h5mu",
    threads: 16
    params:
        model = "neg_binomial",
        model_params = '{"min_confidence": 0.9}',
    log:
        "logs/kaichi/{sample}.log",
    shell:
        "kaichi assign "
        "  --counts {input.counts} "
        "  --guide-library {input.guides} "
        "  --input {input.rna} "
        "  --output {output.h5mu} "
        "  --model {params.model} "
        "  --params '{params.model_params}' "
        "  --threads {threads} "
        "  -v 2> {log}"
```

The pipeline doesn't need any Python or R environment for this step — just the
`kaichi` binary.

---

## Distribution

Binary builds shipped per release for:
- linux x86_64 (musl static; runs on most distros)
- linux aarch64
- macOS x86_64
- macOS arm64
- Windows x86_64 (lower priority)

Plus published to:
- crates.io (`kaichi` binary crate)
- Bioconda (typical bioinformatics distribution channel)
- GitHub Releases (precompiled binaries)
- Homebrew tap (macOS / linuxbrew)

HDF5 is statically linked where possible (`hdf5-metno` or `hdf5` with
`bundled` feature) to avoid system-dep hassles on user machines.

---

## What the CLI does NOT do (yet)

- Differential expression / downstream stats. Out of scope.
- Visualization. Out of scope.
- Cell Ranger replacement. Out of scope (kaichi consumes Cell Ranger output).
- Mixed assignment models across guides in a single run. (Could be added later
  as `--model-per-guide`, but v0 is one model per run.)

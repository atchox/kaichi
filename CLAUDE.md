# CLAUDE.md

Guidance for Claude Code. Architecture, data model, and design rationale live in
[docs/design/](docs/design/README.md) — read those first for context.

---

## What to avoid

- Writing a custom binary format
- Returning JSON or CSV from Rust bindings (kills zero-copy advantage)
- Loom format (older, less ecosystem support)
- Making TileDB-SOMA or Zarr a hard dependency of the Rust core
- Claiming end-to-end zero-copy (copies happen at pandas conversion, HDF5 write, etc.)
- Adding `anndata-rs`, `argmin`, or `polars` as dependencies — not in use, not needed yet

---

## Building

**kaichi-core / kaichi-cli:** `cargo build --release -p kaichi-core -p kaichi-cli`

**kaichi-py:** must be built with `maturin develop` inside `kaichi-py/`, not with
`cargo build`. PyO3's `extension-module` feature requires Python at link time;
plain `cargo build --release` from the workspace root will error on `kaichi-py`.

**kaichi-r:** stub only — no implementation yet.

**Lib name collision:** `kaichi-py` uses `[lib] name = "kaichi_py"` and `kaichi-r`
uses `kaichi_r`. They must stay distinct or Cargo will warn about output filename
collisions in the workspace. Maturin ignores `[lib] name` and uses
`module-name = "kaichi._native"` from `pyproject.toml` for the actual `.so`.

---

## Implementation status

### kaichi-core
- All 11 assignment models implemented and tested
- H5AD read (single-modality) and write implemented
- CSV write implemented
- H5MU read/write: **not yet implemented**
- MEX / Cell Ranger directory input: **not yet implemented**
- molecule_info.h5 UMI dedup path: **not yet implemented**

### kaichi-cli
`kaichi assign --counts <H5AD> --model <NAME> --output <H5AD|CSV>` works.

Designed but not yet implemented:
- `--threads N` (Rayon defaults to all cores — problem on shared HPC nodes)
- `--params <JSON>` (model param overrides; currently hardcoded defaults)
- `--seed N`
- `--guide-library <TSV>`
- `--verbose / --quiet` / structured logging
- `kaichi validate` subcommand
- H5MU output
- MEX input

### kaichi-py
v0.1 complete. `kaichi.assign(h5ad_path, model)` works end-to-end.
`n_jobs` / thread control: **not yet exposed**.

### kaichi-r
Stub only — `Cargo.toml` and an empty `lib.rs`. No implementation yet.

---

## Parallelism

Rayon parallelises the **per-guide EM fitting loop** inside each model's `assign()`.
Each guide's fit is independent — embarrassingly parallel.

Not parallelised:
- Per-cell scoring pass after guide fitting (serial)
- Thread count not exposed — Rayon defaults to all logical CPUs

---

## Known gaps (priority order)

1. `--threads N` CLI + `n_jobs` Python — needed before HPC use
2. `--params` CLI flag — no way to override model defaults from the shell
3. MEX input — CLI design specifies it; currently only H5AD accepted
4. H5MU output + RNA passthrough — primary output format; currently only H5AD/CSV

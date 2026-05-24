# kaichi

[![PyPI](https://img.shields.io/pypi/v/kaichi)](https://pypi.org/project/kaichi/)
[![Python](https://img.shields.io/pypi/pyversions/kaichi)](https://pypi.org/project/kaichi/)
[![License](https://img.shields.io/github/license/atchox/kaichi)](LICENSE)
[![Release](https://img.shields.io/github/actions/workflow/status/atchox/kaichi/release.yml?label=release)](https://github.com/atchox/kaichi/actions/workflows/release.yml)

CRISPR guide assignment for Perturb-seq. Takes a guide-count matrix from Cell Ranger
and produces a per-cell assignment table — which guide each cell received, with
confidence scores and multi-infection flags.

## Interfaces

| Interface | Install | README |
|---|---|---|
| Python (`pip install kaichi`) | PyPI | [kaichi-py/README.md](kaichi-py/README.md) |
| CLI (`kaichi assign`, `kaichi score`, `kaichi decide`) | `cargo install kaichi-cli` | [kaichi-cli/README.md](kaichi-cli/README.md) |

Both interfaces support a **one-shot mode** (`assign`) and a **split mode**
(`score` → `decide`) that fits the model once and lets you re-threshold
without re-running EM.

## Workspace layout

```
kaichi-core/   # Rust library — all 11 models, H5AD I/O
kaichi-cli/    # Command-line interface
kaichi-py/     # Python bindings (PyO3 / maturin)
kaichi-r/      # R bindings (stub, not yet implemented)
```

## Development

```bash
# Rust (core + CLI)
cargo build --release -p kaichi-core -p kaichi-cli
cargo test -p kaichi-core -p kaichi-cli

# Python bindings
pixi install
pixi run build-py
pixi run test-py
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
See [tests/FIXTURES.md](tests/FIXTURES.md) for the reference files and parameter mapping.

## Design documents

[docs/design/](docs/design/README.md) covers architecture, the full model catalog
with algorithm details, I/O spec, and the Python/R binding design.

## Releasing

See [RELEASING.md](RELEASING.md).

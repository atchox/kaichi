# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Two-stage `score` / `decide` API.** `kaichi.score()` runs the EM fitting
  once and returns a cached `ScoreResult`; `kaichi.decide()` applies a
  confidence threshold to the cache and returns a `pyarrow.RecordBatch`.
  Lets users compare multiple thresholds without re-running EM.
- **`kaichi score` and `kaichi decide` CLI subcommands.** Persist the
  `ScoreMatrix` to an H5AD (`X` = preserved UMI counts, `layers/scores` =
  float32 posteriors, `uns/kaichi/stage = "scored"`). `decide` auto-resolves
  the model name and fitted params from `uns/kaichi`, supports `--in-place`,
  and writes a `layers/assigned` group plus assignment `obs` columns.
- **`ScoreMatrix` and `TwoStage` trait in kaichi-core.** Arrow-native CSR
  score matrix (Float32 scores + parallel UInt32 UMI counts) with zero-copy
  PyCapsule exposure.
- **`EmCountMixture` trait.** Deduplicates ~150 lines of identical EM
  orchestration across `poisson`, `neg_binomial`, and `binomial` models;
  responsibilities buffer reused across iterations and restarts.
- **`write_scored_h5ad`, `read_scored_h5ad`, `write_assigned_from_scored`**
  in `kaichi-core::io`. Round-trip tested.
- **Five new regression tests** restored after the EM refactor: NB
  depth-doubles, batch-offset recovery for poisson / neg_binomial /
  binomial, and theta clamp for neg_binomial. Total kaichi-core tests:
  216.
- Python `kaichi.ScoreResult` class re-exported from the native module.
- Python version classifiers for 3.11–3.14 on PyPI.

### Changed

- `AssignmentModel::assign()` for poisson, neg_binomial, binomial, and
  poisson_gauss now delegates to `score()` + `decide()` internally. The
  CLI's existing `kaichi assign` benefits transparently.
- `write_obs` / `write_var` in kaichi-core take `&StringArray` instead of
  `&LoadedInput` so the score writers can reuse them.
- `uns/kaichi` schema now carries `stage` (`"scored"` or `"assigned"`) and
  `min_confidence` (when stage is `"assigned"`).

### Fixed

- Python README incorrectly claimed `assign()` returns `pyarrow.Table`; it
  has always returned `anndata.AnnData`.

## [0.1.0] — 2026-05-23

Initial public release.

### Added

- **kaichi-core**: ten guide-assignment models — `umi`, `max`, `ratio`,
  `gauss`, `poisson_gauss`, `poisson`, `neg_binomial`, `binomial`, `beta2`,
  `beta3`, `quantiles`. Mixture models fit per-guide EM in parallel via
  rayon with multi-restart + best-log-likelihood selection.
- **Hierarchical batch support**: `γ_b` per-batch offset for poisson,
  neg_binomial, binomial; per-batch fitting for beta2/beta3. Reader threads
  `BatchLabels` through from `obs/batch` (categorical or plain string).
- **H5AD single-modality I/O**: read guide-count `X` (handles u32/i64/i32
  on-disk dtypes); write the assignment as `X` + `layers/assigned` +
  `obs[…]` + `uns/kaichi`.
- **CSV write** for lightweight downstream consumption.
- **kaichi-cli**: `kaichi assign --counts <H5AD> --model <NAME>
  --output <H5AD|CSV> [--threads N]`. Thread cap defaults to half of
  logical cores (HPC-polite).
- **kaichi-py (v0.1)**: `kaichi.assign(h5ad_path, model, *, min_confidence,
  quantile, n_jobs)` returns an in-memory `anndata.AnnData` with the
  assignment layered onto preserved UMI counts and provenance in `.uns`.
- Cross-binding **equivalence tests** against
  [crispat](https://github.com/velten-group/crispat) reference outputs for
  all ten models.
- CI/CD release pipeline producing Linux x86_64, macOS arm64, and macOS
  x86_64 wheels (abi3-py311) plus a `cargo install`-able CLI binary.
- Design documents covering architecture, the model catalog with algorithm
  details, I/O spec, and binding interop contracts (`docs/design/`).

[Unreleased]: https://github.com/atchox/kaichi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/atchox/kaichi/releases/tag/v0.1.0

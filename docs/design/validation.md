# kaichi — Validation

kaichi makes a correctness claim that needs verification: the assignment calls are
statistically equivalent to established baseline implementations of the same model
families. This doc defines how that's measured.

**Performance benchmarking is run externally**, via omnibenchmark. kaichi's
responsibility ends at being a well-behaved CLI (stable arguments, sensible exit
codes, deterministic output). The speed targets noted below are design constraints,
not numbers measured inside this repo.

---

## Reference baselines

For each model that has an established baseline implementation in another tool,
kaichi's calls should agree with the baseline **within stochastic tolerance** —
not bit-exact (different optimization paths, different RNG, EM vs SVI), but with
high cell-level agreement on confident calls.

The set of baselines is tracked in `tests/baselines/` as configuration entries:
each entry names a model, a baseline command to run, expected output format
(typically a CSV of `(cell, gRNA, UMI_counts)`), and the kaichi model name it
should be compared to.

Models in the kaichi catalog without a published reference (e.g., specialty
mixtures only implemented in one place) are validated against synthetic
ground-truth only — see below.

---

## Reference datasets

Two tiers:

**Tier 1 — public Perturb-seq data.**

The canonical small/medium fixture is the Schraivogel 2020 TAP-seq dataset, which is
already available locally at the repo's reference-implementation checkout. It ships
with:

- `gRNA_counts.h5ad` — guide count matrix in AnnData form (drops straight into
  kaichi's input path)
- Per-method reference `assignments.csv` files for `umi`, `max`, `ratio`, `gauss`,
  `poisson_gauss`, `poisson`, `neg_binomial`, `binomial`, `2-BetaMM`, `3-BetaMM`,
  `quantiles` — uniform schema `(cell, gRNA, UMI_counts)`
- A second reference set from SCEPTRE on the same input for cross-check

Use this as the **primary equivalence gate** for v0 models. Each model should hit
≥ 99% confident-call agreement with the matching reference CSV before being
considered done.

A larger atlas-scale dataset (≥ 100k cells) can be added later for omnibenchmark.

**Tier 2 — synthetic data generator.** A Rust binary that emits Cell Ranger-format
output with ground-truth guide assignments embedded. Lets us test edge cases that
real data rarely covers cleanly: high ambient, very low UMI depth, rare guides,
extreme multi-infection rates. Particularly important for the NB / mixture models —
synthetic with known generating parameters is the cleanest way to verify EM recovers
the truth.

---

## Equivalence tests

### Cross-language equivalence (Python ↔ R)

The most fundamental guarantee: **the same input produces the same calls regardless
of which binding the user calls through.** Test setup:

1. Fixture: one small reference dataset.
2. Run `kaichi-py::assign_guides` and `kaichi-r::assign_guides` with identical params.
3. Compare Arrow record batches before any language-native object construction.

Tolerance: bit-exact for `threshold` / `max` / `ratio` / `umi`;
float-equal within `1e-6` for EM-based models (both bindings call the same Rust
code so this should hold trivially, but verify).

### Baseline agreement

Per model, on each reference dataset:

1. Run the baseline implementation with default params, save calls.
2. Run kaichi with matching params, save calls.
3. Report:
   - **Confident-call agreement** — for cells where both tools assign (no unassigned),
     fraction of cells with identical `guide_id`. Target ≥ 99% on Tier 1 datasets.
   - **Confidence correlation** — Spearman correlation of `assignment_confidence`
     across cells. Target ≥ 0.95.
   - **Unassigned-rate agreement** — fraction unassigned should be within 1% absolute.

Disagreements are flagged for manual inspection — model differences are sometimes real
and sometimes bugs.

### Ground-truth recovery (synthetic)

For synthetic data with known assignments:
- Precision and recall per model
- ROC curves for `assignment_confidence`
- Behavior under varying ambient noise and UMI depth

---

## Performance target (design constraint)

**≥ 20× speedup over existing pyro-based implementations on 10k cells × 1k guides,
single workstation, no GPU.** Stretch: ≥ 100× when per-guide parallelism is fully
utilized (16+ cores).

This is a design constraint, not a measurement performed in this repo. omnibenchmark
runs the actual comparisons; kaichi only needs to expose a stable CLI for it to
invoke (see [cli.md](cli.md)).

---

## Numerical reproducibility

EM is deterministic given initialization. Two design choices:

1. **Init from a fixed seed.** k-means initialization of mixture components uses a
   seed surfaced as a model parameter. Same input + same seed + same params →
   identical output. Recorded in `uns['kaichi']['model_params']['seed']`.
2. **rayon does not affect determinism.** Per-guide fits are independent; the only
   parallelism risk is non-deterministic reduction order, which we avoid by keeping
   per-guide fits independent (no cross-guide reductions in the fitting loop).

---

## What "ship" requires

Before kaichi is considered ready for users:

1. `umi`, `max`, `ratio`, `neg_binomial`, and `poisson_gauss` pass baseline
   equivalence on all Tier 1 datasets.
2. Cross-language equivalence (Python ↔ R) passes for all v0 models on all fixtures.
3. Synthetic ground-truth recovery within published expectations for each model.
4. CLI is stable enough for omnibenchmark integration (documented args, stable
   exit codes, deterministic output for fixed seed).

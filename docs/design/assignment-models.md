# kaichi — Assignment Models

All models read the same input (a cells × guides UMI count matrix, plus per-cell
covariates) and produce the same Arrow schema output
(see [data-model.md](data-model.md)). They differ in **what information they use**
and **how they decide "assigned" vs "unassigned" vs "multi-infected"**.

## Method catalog

Eleven models, grouped by what information they use during assignment:

| Category | Models |
|---|---|
| **Independent (per-cell, per-guide, threshold)** | `umi` |
| **Across gRNAs (within a cell)** | `max`, `ratio` |
| **Across cells (per-guide)** | `gauss`, `poisson_gauss` |
| **Across gRNAs and cells (per-guide hierarchical mixture)** | `beta2`, `beta3`, `poisson`, `neg_binomial`, `binomial`, `quantiles` |

kaichi v0 targets the speed-critical subset first; the rest are stubbed against the
trait and filled in over time:

| Priority | Models | Notes |
|---|---|---|
| **v0** | `umi`, `max`, `ratio`, `neg_binomial`, `poisson_gauss` | Trivial baselines + the two hottest hierarchical models |
| **v0.1** | `gauss`, `poisson`, `binomial` | Remaining across-cell and hierarchical models |
| **v0.2** | `beta2`, `beta3`, `quantiles` | Specialty models |

---

## Algorithm choice: EM, not SVI

Hierarchical models (NB, Poisson, Beta, Binomial mixtures) are commonly fitted by
**pyro / SVI** — stochastic variational inference with Adam over thousands of
subsampled gradient steps. For the per-guide problem size, this is the wrong tool:

- **Autodiff overhead dominates the actual math.** Per-step PyTorch graph
  construction costs more than the gradient itself for problems this small.
- **The posterior is unimodal and well-behaved.** Closed-form EM converges in
  10–50 iterations with deterministic updates. SVI needs thousands of noisy
  gradient steps for an approximate posterior we don't actually need.
- **Subsampling is a workaround for autodiff cost.** SVI-based implementations
  subsample cells per step to make autodiff tractable. EM doesn't need to subsample —
  the M-step is a weighted MLE pass over all data per iteration.
- **Per-gRNA parallelism is embarrassingly easy.** rayon over gRNAs replaces
  cluster-based parallelism (Dask etc.) used by some implementations.

kaichi implements these model families (same likelihoods, same priors, same
covariates) via EM, with `argmin` for any inner Newton-Raphson steps required for
non-closed-form M-steps (e.g., NB MLE). **Same model, different algorithm.**

---

## Required per-cell covariates

The hierarchical models take two per-cell covariates kaichi must support:

```
batch         categorical    e.g., experimental run / pool / library prep batch
total_counts  Float32        per-cell sequencing depth (sum of guide UMIs)
```

These come from `.obs` of the input AnnData / MuData. `total_counts` is computed
by kaichi from `.X` if not provided. `batch` defaults to a single batch if not
provided.

---

## Model: `umi` (independent per-cell threshold)

**Assign each cell-guide pair as positive if UMI count ≥ threshold.**

The simplest baseline.

```
Parameters:
  umi_threshold   UInt32   default: 5
```

Per cell:
- 0 guides above threshold → `is_unassigned`
- 1 guide above threshold → assign
- 2+ guides above threshold → `is_multi_infected`; assign to highest UMI

---

## Model: `max`

**Assign each cell to the guide with the highest UMI count.**

```
Parameters:
  umi_threshold   UInt32   default: 0    (optional post-hoc filter)
```

Per cell:
- All-zero counts → `is_unassigned`
- Single highest guide → assign
- Multiple guides tied for highest → `is_unassigned`

---

## Model: `ratio`

**Assign each cell to the guide with the largest fraction of total guide UMIs, if that fraction exceeds a threshold.**

```
Parameters:
  min_fraction    Float32  default: 0.3
```

Per cell:
- All-zero total UMI → `is_unassigned`
- `top_fraction = top_guide_umi / sum(all_guide_UMIs_for_cell)`
- `top_fraction > min_fraction` → assign top guide; `assignment_confidence = top_fraction`
- Otherwise → `is_unassigned`

`is_multi_infected` is not set by this model — cells whose dominant guide does not clear the threshold are simply unassigned.

---

## Model: `neg_binomial` (the workhorse)

**Per-guide negative binomial mixture with per-cell batch and depth covariates.**
This is the speed-critical model. SCEPTRE-style mixture: cells either carry the
guide (signal) or do not (background), and observed UMI counts follow a Negative
Binomial whose mean depends on the latent perturbation state and per-cell covariates.

### Model

For each guide g and each cell i:

```
z_{i,g} ~ Bernoulli(π_g)                              latent: 1 if cell carries guide g
log(μ_{i,g} | z) = β0_g + β1_g · z + γ_{batch(i),g} + log(total_counts_i)
y_{i,g} ~ NegativeBinomial(μ_{i,g}, θ_g)              observed UMI count
```

Per-guide global parameters: `β0_g` (baseline), `β1_g` (perturbation effect on log-mean),
`γ_{b,g}` (batch effects), `π_g` (perturbation rate), `θ_g` (dispersion).

### Parameters

```
min_confidence  Float32   default: 0.8     posterior P(z=1) threshold for assignment
max_em_iters    UInt32    default: 100
tol             Float32   default: 1e-6    log-likelihood convergence
init_seed       UInt64    default: 2024    determinism
min_nonzero     UInt32    default: 2       skip guides with fewer cells > 0
min_max_count   UInt32    default: 2       skip guides whose max count is below this
```

`min_nonzero` and `min_max_count` are the per-guide skip rules: a guide with only a
single non-zero cell, or whose maximum count is below 2, has too little data to fit a
2-component mixture and is left unassigned for all cells.

### Fitting (EM)

Per guide (parallelized via `rayon::par_iter`):

1. **Initialize**: k-means with k=2 on log(UMI + 1) over non-zero cells, or threshold-
   based init at the 90th percentile. Init `β0` to log-mean of low component,
   `β1` so β0+β1 ≈ log-mean of high component, `θ` = 5.0, `π` = 0.01, `γ` = 0.
2. **E-step**: compute responsibilities `r_i = P(z_i = 1 | y_i, params)` via Bayes rule
   on the two NB component PMFs. Closed form.
3. **M-step**:
   - `π_new` = mean of responsibilities
   - `(β0, β1, γ, θ)`: weighted MLE for NB GLM. No closed form (digamma involved).
     Use Newton-Raphson via `argmin`. Typically converges in <10 inner iterations.
4. **Convergence**: stop when ΔlogLik / |logLik| < `tol` or `max_em_iters` reached.

### Assignment

For each cell, compute posterior P(z=1) under fitted params for each guide. Then:
- 0 guides with posterior ≥ `min_confidence` → `is_unassigned`
- 1 guide → assign; `assignment_confidence` = that posterior
- 2+ guides → `is_multi_infected`; assign to max-posterior guide

### Performance properties

- No autodiff. NB log-likelihood derivatives are derived once.
- No subsampling. M-step uses all cells per iteration.
- ~50 iterations to convergence vs the thousands typical of SVI.
- rayon over guides; no cluster overhead.

Target: ≥ 20× speedup over existing pyro-based implementations on 10k cells × 1k
guides, single workstation, no GPU.

### Implementation notes (numerical care required)

The NB M-step has no closed form. Newton-Raphson with hand-derived gradients/Hessians
of the NB log-likelihood — involving digamma and trigamma functions for the dispersion
parameter — is the standard approach. Easy to subtly mis-implement.

Pitfalls to guard against:
- **Dispersion `θ → 0` or `θ → ∞`** (over- or under-dispersion limits). Clamp or
  reparameterize as log-θ.
- **All-zero guides** (a guide present in the library but never detected). Already
  handled by the `min_nonzero` / `min_max_count` skip rules above; verify in tests.
- **Single-cell guides** (only one cell has any UMI for this guide). Skip — too little
  data to fit a 2-component mixture.
- **Degenerate batch effects** (a batch with no cells, or all cells in one batch).
  Drop empty batches before fitting; verify γ identifiability.
- **Convergence stalls.** Cap iterations at `max_em_iters`; report non-convergence in
  the per-guide diagnostics rather than failing the whole run.

Recommended development sequence for this model specifically:

1. **Write the EM test before the EM code.** Generate synthetic 2-component NB with
   known parameters (β0, β1, θ, π, γ); fit; assert recovery within tolerance.
2. **Validate against a reference fixture early.** See [validation.md](validation.md)
   for the Schraivogel-derived equivalence test — agreement should be ≥ 99% before
   considering this model done.

---

## Model: `poisson` and `poisson_gauss`

`poisson`: same architecture as `neg_binomial` but the count likelihood is Poisson
(no dispersion parameter `θ`). Faster, but underfits overdispersed counts.
Closed-form weighted Poisson MLE is available — even faster M-step than NB.

`poisson_gauss`: per-guide mixture where background ~ Poisson(λ) and signal ~
N(μ, σ²), both on **raw UMI counts**. Zeros are included analytically in the EM
updates as Poisson background observations — no log transform, no zero exclusion.
Both components have closed-form M-steps (Poisson MLE = weighted mean of counts;
Gaussian MLE = weighted mean and variance). This is the same statistical model as
crispat's pgmm; the difference is EM (kaichi) vs SVI (crispat).

Implemented parameters:

```
min_confidence  Float32  default: 0.5
max_em_iters    UInt32   default: 100
tol             Float32  default: 1e-6
min_nonzero     UInt32   default: 2
min_max_count   UInt32   default: 2
```

Assignment follows the same posterior-threshold pattern as `neg_binomial`: each
guide is fit independently, cells with posterior signal probability above
`min_confidence` are candidates, and cells with multiple passing guides are marked
multi-infected.

---

## Model: `gauss`

Per-guide 2-component Gaussian mixture on log(UMI + 1). No covariates.
Fully closed-form EM.

Parameters:
```
min_confidence  Float32  default: 0.8
min_umi         UInt32   default: 1
```

---

## Model: `binomial`

Per-guide binomial mixture: signal cells have higher P(observed | total_counts),
background cells lower. EM with closed-form M-step (binomial weighted MLE).

---

## Model: `beta2` and `beta3`

2- or 3-component beta mixtures on a transformed quantity (guide proportion within
total guide counts). EM with numerical M-step (no closed form for beta MLE —
method-of-moments or Newton-Raphson via `argmin`).

Lower priority — defer detail until v0.2 implementation begins.

---

## Model: `quantiles`

Quantile-based assignment. Not a mixture model in the same sense — assigns based
on cell-rank percentile within each guide's UMI distribution. Spec to be filled
in during implementation.

---

## Choosing a model

Defaults:

| Scenario | Recommended model |
|---|---|
| Quick sanity check, clean data | `umi` |
| Standard CRISPRi/a with simple cells | `max` |
| Clear bimodality concern | `ratio` |
| Multi-batch, variable sequencing depth | `neg_binomial` |
| Lower runtime than NB, simpler dispersion | `poisson_gauss` |

The model name is recorded in `uns['kaichi']['model']` so calls from different
models can be compared on the same input.

---

## Implementation contract (Rust)

```rust
pub trait AssignmentModel: Send + Sync {
    fn name(&self) -> &'static str;
    fn assign(&self, counts: &RecordBatch, covariates: &CovariateFrame)
        -> Result<RecordBatch>;
    fn params_json(&self) -> serde_json::Value;
}
```

- `counts`: Arrow schema `(cell_barcode: Utf8, guide_id: Utf8, umi_count: UInt32)`.
  Conceptually long-form, but the implementation typically converts to a sparse CSR
  internally.
- `covariates`: per-cell batch (categorical) and total_counts (Float32).
- Output: schema in [data-model.md](data-model.md).
- `params_json`: serialized parameters for `uns['kaichi']['model_params']`.

Models require guide IDs and counts, not guide-library metadata. If a guide library is
provided, metadata enrichment happens before writing or object construction so fields
such as `target_gene` can be populated. Without a guide library, assignments remain
valid at the `guide_id` level and annotation fields stay null or absent.

Per-guide work happens inside `assign`. Models are free to use `rayon::par_iter`
over guides internally; the trait is `Send + Sync` to allow callers to share models
across threads.

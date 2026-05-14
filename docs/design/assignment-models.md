# kaichi — Assignment Models

All models read the same input (a cells × guides UMI count matrix, plus per-cell
covariates) and produce the same Arrow schema output
(see [data-model.md](data-model.md)). They differ in **what information they use**
and **how they decide "assigned" vs "unassigned" vs "multi-infected"**.

## Method catalog

Eleven models, grouped by what information they use during assignment:

| Category | Models |
|---|---|
| **Per-cell threshold (no fitting)** | `umi`, `max`, `ratio` |
| **Per-guide mixture, no covariates** | `gauss`, `poisson_gauss` |
| **Per-guide mixture with batch + depth covariates (SCEPTRE-style)** | `poisson`, `neg_binomial`, `binomial` |
| **Per-batch mixture on cell-wise max proportion** | `beta2`, `beta3` |
| **Rank-based (no fitting)** | `quantiles` |

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

## EM convention used by all mixture models

The hierarchical models below all share the same per-guide (or per-batch) two-component
mixture structure. Define for cell `i`:

- `y_i`: observed UMI count (or proportion, for `beta2`/`beta3`)
- `z_i ∈ {0, 1}`: latent perturbation state (0 = background, 1 = signal)
- `r_i = P(z_i = 1 | y_i, θ)`: posterior signal responsibility under current parameters

The EM loop is:

1. **E-step**: compute `r_i` via Bayes rule on the two component likelihoods.
   `r_i = π·f₁(y_i) / (π·f₁(y_i) + (1-π)·f₀(y_i))`
2. **M-step**: maximize the expected complete-data log-likelihood with respect to the
   model parameters, treating `r_i` as fixed weights. Each cell contributes a
   weighted background term and a weighted signal term.
3. **Convergence**: stop when `|ΔlogLik| / max(|logLik|, 1) < tol` or
   `max_em_iters` reached.

**Identifiability** is enforced by labeling components after fitting: the component with
larger mean / proportion / location is the **signal** component. We do not constrain
during fitting (constrained optimization slows convergence); we just relabel post-fit.

**Skip rules** (shared across all mixture models): if a guide has fewer than
`min_nonzero` non-zero cells, or its maximum count is below `min_max_count`, the
guide is skipped — no fit attempted, all cells unassigned for that guide. Defaults
`min_nonzero = 2`, `min_max_count = 2` follow crispat.

**Zero handling**: crispat fits SVI mixture models on **non-zero counts only** (it
discards zeros before fitting). kaichi includes zeros analytically wherever the
likelihood admits a closed-form contribution from a single shared zero value (the
Poisson, Poisson-Gauss, and Binomial families). This removes the truncated-likelihood
bias from crispat's `λ`/`p` estimates without materializing the zero observations. See
[poisson_gauss.rs](../../kaichi-core/src/models/poisson_gauss.rs) for the pattern.

---

## Model: `gauss`

**Per-guide 2-component Gaussian mixture on log10(UMI + 1).** No covariates.
Same architecture as crispat's `ga_gauss` with `inference="em"` (which delegates
to `sklearn.mixture.GaussianMixture(n_components=2, covariance_type="tied")`).

### Model

For each guide and each cell `i`:

```
x_i = log10(y_i + 1)
x_i | z_i = 0 ~ N(μ_l, σ²)         background
x_i | z_i = 1 ~ N(μ_h, σ²)         signal (μ_h > μ_l)
P(z_i = 1) = π
```

**Tied variance** `σ²` is shared between components, matching crispat's
`covariance_type="tied"`. Crispat fits per **batch** (one mixture per batch over all
guides); kaichi can do per-batch or per-guide — controlled by the `scope` parameter.
Default is per-guide because it allows guides with very different baselines to be
modeled separately.

### Parameters

```
min_confidence  Float32   default: 0.8     posterior P(z=1) threshold for assignment
max_em_iters    UInt32    default: 100
tol             Float32   default: 1e-6
min_nonzero     UInt32    default: 2
min_max_count   UInt32    default: 2
nonzero_only    bool      default: false   if true, fit on x_i > 0 (drops zero cells)
scope           enum      default: guide   "guide" or "batch"
```

`nonzero_only = false` (kaichi default) includes zero counts — they map to
`x_i = log10(1) = 0`, which the model can fit as the bulk of the background. crispat's
SVI default is also `nonzero=False`, but its EM path supports both.

### EM updates (closed form)

Initialize `π = 0.01`, `μ_l = 0`, `μ_h = 1`, `σ = 1`. With per-cell responsibility `r_i`:

```
E-step:
  r_i = π·φ(x_i; μ_h, σ) / (π·φ(x_i; μ_h, σ) + (1-π)·φ(x_i; μ_l, σ))

M-step:
  π_new   = (1/N) · Σ_i r_i
  μ_h_new = Σ_i r_i · x_i / Σ_i r_i
  μ_l_new = Σ_i (1-r_i) · x_i / Σ_i (1-r_i)
  σ²_new  = (1/N) · Σ_i [r_i · (x_i - μ_h_new)² + (1-r_i) · (x_i - μ_l_new)²]
```

All updates are closed form. No inner Newton-Raphson.

### Assignment

For each cell, compute `r_i` under the fitted parameters. Posterior threshold logic
matches crispat:

- `r_i ≥ min_confidence` for ≥ 1 guide → assign to max-posterior guide
- `r_i ≥ min_confidence` for ≥ 2 guides → also flag `is_multi_infected`
- Otherwise → `is_unassigned`

---

## Model: `poisson_gauss`

**Per-guide mixture where background ~ Poisson(λ) and signal ~ N(μ, σ²), both on
raw UMI counts.** Same statistical model as crispat's `ga_poisson_gauss`; the
difference is EM (kaichi) vs SVI (crispat).

### Model

For each guide and each cell `i`:

```
y_i | z_i = 0 ~ Poisson(λ)                background
y_i | z_i = 1 ~ N(μ, σ²)                  signal
P(z_i = 1) = π
```

Zeros are included analytically as Poisson background observations — no log
transform, no zero exclusion. See the [implementation](../../kaichi-core/src/models/poisson_gauss.rs)
for the analytic-zero pattern.

### Parameters

```
min_confidence  Float32  default: 0.5
max_em_iters    UInt32   default: 100
tol             Float32  default: 1e-6
min_nonzero     UInt32   default: 2
min_max_count   UInt32   default: 2
```

### EM updates (closed form)

Let `n_z` be the number of zero-count cells, `r_z` the (shared) responsibility for a
zero cell at the current parameters, and let `i` range over the **non-zero** cells
with responsibility `r_i`. Then:

```
E-step:
  r_i = π·N(y_i; μ, σ) / (π·N(y_i; μ, σ) + (1-π)·Pois(y_i; λ))
  r_z = π·N(0; μ, σ)  / (π·N(0; μ, σ)  + (1-π)·Pois(0; λ))

M-step:
  S_r       = Σ_i r_i + n_z · r_z
  S_1_minus = (N_nonzero - Σ_i r_i) + n_z · (1 - r_z)
  π_new     = S_r / N_total
  λ_new     = Σ_i (1-r_i) · y_i / S_1_minus            (zeros contribute 0 to numerator)
  μ_new     = Σ_i r_i · y_i / S_r                      (zeros contribute 0)
  σ²_new    = (Σ_i r_i · (y_i - μ)² + n_z · r_z · μ²) / S_r
```

The `n_z · r_z · μ²` term is the contribution of zero cells to the signal variance:
each zero contributes `(0 - μ)² = μ²`.

### Implementation status

✅ Implemented. See [poisson_gauss.rs](../../kaichi-core/src/models/poisson_gauss.rs)
and its unit tests.

---

## Model: `poisson` (SCEPTRE)

**Per-guide Poisson mixture with batch and depth covariates.** SCEPTRE-style:
log-mean is a linear function of latent state, batch, and per-cell sequencing depth.

### Model

For each guide and each cell `i` with batch `b(i)` and total guide UMIs `d_i`:

```
z_i ~ Bernoulli(π)
log(μ_i) = β0 + β1·z_i + γ_{b(i)} + log(d_i + 1)
y_i ~ Poisson(μ_i)
```

Per-guide parameters: `β0` (baseline log-rate), `β1 > 0` (perturbation effect),
`γ_b` (per-batch offset, identifiable up to a global shift absorbed into β0),
`π` (perturbation rate). `d_i` is the per-cell total guide UMI count (a covariate,
not a parameter).

### Parameters

```
min_confidence  Float32   default: 0.8
max_em_iters    UInt32    default: 100
inner_max_iters UInt32    default: 25     Newton-Raphson on β,γ per M-step
tol             Float32   default: 1e-6
min_nonzero     UInt32    default: 2
min_max_count   UInt32    default: 2
```

### EM updates

Let `μ_i^0 = exp(β0 + γ_{b(i)} + log(d_i+1))`, `μ_i^1 = exp(β0 + β1 + γ_{b(i)} + log(d_i+1))`.
Crispat fits on non-zero cells only; kaichi can include zeros (they contribute the
single-shared-zero analytic terms below).

```
E-step:
  r_i = π·Pois(y_i; μ_i^1) / (π·Pois(y_i; μ_i^1) + (1-π)·Pois(y_i; μ_i^0))

M-step:
  π_new = mean(r_i)

  (β0, β1, γ) by Newton-Raphson on the weighted log-likelihood:
    Q(β,γ) = Σ_i [ r_i·(y_i·log μ_i^1 - μ_i^1) + (1-r_i)·(y_i·log μ_i^0 - μ_i^0) ]

    ∂Q/∂β0 = Σ_i [r_i·(y_i - μ_i^1) + (1-r_i)·(y_i - μ_i^0)]                = 0
    ∂Q/∂β1 = Σ_i r_i·(y_i - μ_i^1)                                          = 0
    ∂Q/∂γ_b = Σ_{i ∈ b} [r_i·(y_i - μ_i^1) + (1-r_i)·(y_i - μ_i^0)]         = 0
```

The Hessian is block-diagonal in the batches (each `γ_b` only couples to itself
and `β0`/`β1`). Newton-Raphson converges in 3–8 inner iterations. Use `argmin` with
hand-derived gradient/Hessian — no autodiff.

### Identifiability

`β1 > 0` is enforced post-fit by relabeling: if Newton-Raphson lands at `β1 < 0`,
swap the two component definitions (the math is symmetric in {z=0, z=1}). To avoid
sign flips during fitting, initialize `β0` near the log-mean of low-count cells and
`β1 = log(max_count / median_nonzero)`, then keep `β1` unconstrained — relabeling at
convergence is cheaper than imposing a constraint.

### Crispat divergences

- **Zero handling**: crispat drops `y_i = 0` cells before fitting. kaichi includes them
  via the same single-shared-zero pattern as `poisson_gauss`. This eliminates the
  truncated-Poisson bias in `λ`/`β0` estimation.
- **Subsampling**: crispat subsamples 15k cells per SVI step. kaichi uses all cells
  per M-step (cheaper because no autodiff).

---

## Model: `neg_binomial` (the workhorse)

**Per-guide Negative Binomial mixture with batch and depth covariates.** Same
architecture as `poisson` plus an overdispersion parameter `θ`. This is the
speed-critical model and the one closest to SCEPTRE's published form.

### Model

For each guide and each cell `i`:

```
z_i ~ Bernoulli(π)
log(μ_i) = β0 + β1·z_i + γ_{b(i)} + log(d_i + 1)
y_i ~ NegativeBinomial(μ_i, θ)
  with mean = μ_i and variance = μ_i + μ_i²/θ
```

Per-guide parameters: `β0`, `β1 > 0`, `γ_b`, `π`, `θ > 0` (dispersion). As `θ → ∞`
the NB collapses to Poisson; smaller `θ` means more overdispersion.

### Parameters

```
min_confidence  Float32   default: 0.8
max_em_iters    UInt32    default: 100
inner_max_iters UInt32    default: 25
tol             Float32   default: 1e-6
min_nonzero     UInt32    default: 2
min_max_count   UInt32    default: 2
theta_init      Float32   default: 5.0
theta_min       Float32   default: 0.01    clamp to avoid θ → 0 degeneracy
theta_max       Float32   default: 1e4     clamp to avoid θ → ∞ degeneracy
```

### EM updates

```
E-step:
  r_i = π·NB(y_i; μ_i^1, θ) / (π·NB(y_i; μ_i^1, θ) + (1-π)·NB(y_i; μ_i^0, θ))

M-step:
  π_new = mean(r_i)

  (β0, β1, γ, θ): joint Newton-Raphson on the NB weighted log-likelihood.
  Standard derivatives — log Γ(y+θ), digamma ψ(·), trigamma ψ'(·) — apply.
  Reparameterize θ as φ = log θ for unconstrained optimization; clamp φ to
  [log θ_min, log θ_max] each inner iteration.
```

Inner Newton-Raphson handled by `argmin` with hand-derived gradient and Hessian.

### Numerical pitfalls

- **`θ → ∞`** when the data are exactly Poisson. The likelihood becomes flat; clamp.
- **`θ → 0`** when one component overfits a single outlier. Clamp.
- **Empty batches** after subsetting to a single guide's nonzero cells. Drop empty
  batches before assembling the design matrix.
- **Single-cell guides** are skipped by `min_nonzero`.
- **Convergence stalls**: if `θ` oscillates between iterations, halve the Newton step.

### Performance target

≥ 20× speedup over crispat's `ga_negative_binomial` on Replogle K562 essential
(2,057 guides × ~620k cells), single workstation, no GPU. See [validation.md](validation.md)
for benchmarking protocol.

---

## Model: `binomial`

**Per-guide Binomial mixture using the total guide UMI count as the number of trials.**

### Model

For each guide and each cell `i` with batch `b(i)` and total guide UMIs `d_i`:

```
z_i ~ Bernoulli(π)
logit(p_i) = β0 + β1·z_i + γ_{b(i)}
y_i ~ Binomial(d_i, p_i)
```

Per-guide parameters: `β0`, `β1 > 0`, `γ_b`, `π`. The trial count `d_i` is the
per-cell **total guide UMIs**, not the total RNA — this is what crispat passes.

**Note on a crispat inconsistency**: crispat's `binomial.py` model defines
`p = sigmoid(β0 + β1·z + γ_batch)` (correct), but `get_perturbed_cells` evaluates
`sigmoid(exp(β0 + γ_batch))` (always > 0.5, almost certainly a bug — this would give
`p` near 1 for background cells). kaichi follows the model definition.

### Parameters

```
min_confidence  Float32   default: 0.8     posterior threshold for assignment
max_em_iters    UInt32    default: 100
inner_max_iters UInt32    default: 25
tol             Float32   default: 1e-6
min_nonzero     UInt32    default: 2
min_max_count   UInt32    default: 2
```

### EM updates

```
E-step:
  r_i = π·Binom(y_i; d_i, p_i^1) / (π·Binom(y_i; d_i, p_i^1) + (1-π)·Binom(y_i; d_i, p_i^0))

M-step:
  π_new = mean(r_i)

  (β0, β1, γ): IRLS on the weighted logistic-regression log-likelihood
    Q(β,γ) = Σ_i [ r_i·logBinom(y_i; d_i, p_i^1) + (1-r_i)·logBinom(y_i; d_i, p_i^0) ]
  Standard logistic IRLS converges in 3–6 inner iterations.
```

The trial count `d_i` is fixed per cell — it only appears in the log-binomial
coefficient (which doesn't depend on `β,γ`) and in the mean `d_i·p_i` (which does).
So `∂Q/∂β` involves only `(y_i - d_i·p_i)·∂p_i/∂β` terms — same form as standard
logistic regression with offset.

Zero cells (`y_i = 0`) can be included analytically: their contribution to `r_i`
depends only on `p_i^0` and `p_i^1`, both of which are functions of `β`/`γ` and not of
`y_i` (for `y=0`, `logBinom = d_i · log(1-p_i)` plus a constant). The "single shared
zero" trick doesn't apply directly because `d_i` varies; if performance matters,
batch the zeros by `(b(i), d_i)` and weight by the count.

---

## Model: `beta2`

**2-component Beta mixture on the maximum guide proportion per cell.** Per-batch
fitting (one mixture per batch, applied to all guides simultaneously). Matches
crispat's `ga_2beta`.

### Model

For each cell `i` in batch `b`:

```
x_i = max_g (y_{i,g} / Σ_g y_{i,g})        max guide proportion, clamped to (1e-4, 0.9999)
z_i ∈ {0, 1}                                background / signal
x_i | z_i = 0 ~ Beta(α_l, β_l)
x_i | z_i = 1 ~ Beta(α_h, β_h)
P(z_i = 1) = π
```

The signal component has mean `α_h / (α_h + β_h)` close to 1 (one guide dominates);
the background has mean closer to `1/G` (counts spread across many guides).

### Parameters

```
min_confidence  Float32  default: 0.5     posterior threshold
max_em_iters    UInt32   default: 200
tol             Float32  default: 1e-6
clamp_lo        Float32  default: 1e-4    Beta is undefined at 0
clamp_hi        Float32  default: 1-1e-4
```

### EM updates

```
E-step:
  r_i = π·Beta(x_i; α_h, β_h) / (π·Beta(x_i; α_h, β_h) + (1-π)·Beta(x_i; α_l, β_l))

M-step:
  π_new = mean(r_i)

  (α_k, β_k) for k ∈ {l, h} have no closed-form Beta MLE.
  Two-step update per component:
    1. Method of moments (closed form):
        x̄_k = Σ_i w_{ik}·x_i / Σ_i w_{ik}        (weighted mean)
        v_k = Σ_i w_{ik}·(x_i - x̄_k)² / Σ_i w_{ik}  (weighted variance)
        φ_k = x̄_k·(1-x̄_k) / v_k - 1
        α_k = x̄_k · φ_k,   β_k = (1 - x̄_k) · φ_k
       where w_{i,h} = r_i and w_{i,l} = 1 - r_i.
    2. Optional Newton-Raphson refinement on digamma equations (1–2 inner steps
       are usually enough). Skip if speed matters more than fit quality.
```

Method-of-moments is good enough for the mixture E-step to make progress; the
refined fit emerges over EM iterations.

### Initialization

Crispat uses `α = [1, 10]`, `β = [10, 1]`, mirroring a low-mean and high-mean Beta.
kaichi uses the same: `(α_l, β_l) = (1, 10)`, `(α_h, β_h) = (10, 1)`, `π = 0.4`.

---

## Model: `beta3`

**3-component Beta mixture on the maximum guide proportion per cell.** Same shape
as `beta2`, with an additional intermediate component (cells with some signal but
not dominant). Matches crispat's `ga_3beta`.

### Model

```
x_i ~ π_l · Beta(α_l, β_l)  +  π_m · Beta(α_m, β_m)  +  π_h · Beta(α_h, β_h)
```

with `π_l + π_m + π_h = 1` and components ordered by mean (`l < m < h`).

### EM updates

Same as `beta2` extended to three components:

```
E-step:
  r_{ik} = π_k · Beta(x_i; α_k, β_k) / Σ_j π_j · Beta(x_i; α_j, β_j)

M-step:
  π_k_new = mean(r_{ik})
  (α_k, β_k): method of moments per component (closed form), optionally refined.
```

Only the `h` (high) component is treated as signal for assignment. The `m`
(intermediate) component is used purely to fit non-canonical cells without
contaminating the `l`/`h` estimates — same intent as crispat.

### Identifiability

Sort components by mean post-fit. Crispat's init `α = [1, 10, 10]`, `β = [10, 10, 1]`
makes `l` the low-mean, `m` the middle, `h` the high-mean. Use the same.

---

## Model: `quantiles`

**Top-X% cells per guide by guide-proportion rank.** Not a mixture model — no EM,
no parameters except the quantile threshold(s). Matches crispat's `ga_quantiles`.

### Algorithm

For each guide `g`:

1. Compute `p_{i,g} = y_{i,g} / Σ_g' y_{i,g'}` for every cell with non-zero
   total guide UMIs and `y_{i,g} > 0`.
2. Sort cells by `p_{i,g}` descending.
3. Top `⌊quantile · N_g⌋` cells (where `N_g` is the number of cells with
   `y_{i,g} > 0`) are assigned to guide `g`.

### Parameters

```
quantiles   List<Float32>   no default       e.g., [0.01, 0.05, 0.10]
```

Crispat takes a **list** of thresholds and writes one assignment CSV per threshold.
kaichi follows the same — a single run can produce multiple H5AD outputs at
different quantiles, one file per threshold, since the per-guide sort is reused.

### When to use this

This is a calibration tool, not a model. The use case is: you have a rough
expectation of MOI (multiplicity of infection) and want a conservative cell list
without committing to a parametric model. Useful for downstream QC and for sanity-
checking mixture-model outputs.

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

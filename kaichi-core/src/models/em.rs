use std::f64::consts::PI;
use statrs::function::gamma::ln_gamma;

/// Numerically stable log(exp(a) + exp(b)).
#[inline]
pub fn logsumexp2(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Clamp probability to (ε, 1−ε) to prevent log(0) in EM updates.
#[inline]
pub fn clamp_probability(v: f64) -> f64 {
    v.clamp(1e-6, 1.0 - 1e-6)
}

/// Log-density of N(x; mean, sigma).
#[inline]
pub fn log_normal_pdf(x: f64, mean: f64, sigma: f64) -> f64 {
    let z = (x - mean) / sigma;
    -0.5 * (2.0 * PI).ln() - sigma.ln() - 0.5 * z * z
}

/// Log-PMF of Poisson(x; lambda).
#[inline]
pub fn log_poisson_pmf(x: f64, lambda: f64) -> f64 {
    x * lambda.ln() - lambda - ln_gamma(x + 1.0)
}

/// Log-PMF of NegBin(y; mu, theta) with mean=mu, variance=mu+mu²/theta.
#[inline]
pub fn log_nb_pmf(y: f64, mu: f64, theta: f64) -> f64 {
    ln_gamma(y + theta) - ln_gamma(theta) - ln_gamma(y + 1.0)
        + theta * (theta / (theta + mu)).ln()
        + y * (mu / (mu + theta)).ln()
}

/// Log-PMF of Binomial(y; n, p).
#[inline]
pub fn log_binom_pmf(y: f64, n: f64, p: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(y + 1.0) - ln_gamma(n - y + 1.0)
        + y * p.ln()
        + (n - y) * (1.0 - p).ln()
}

/// Log-PDF of Beta(x; alpha, beta_).
#[inline]
pub fn log_beta_pdf(x: f64, alpha: f64, beta_: f64) -> f64 {
    (alpha - 1.0) * x.ln() + (beta_ - 1.0) * (1.0 - x).ln()
        - (ln_gamma(alpha) + ln_gamma(beta_) - ln_gamma(alpha + beta_))
}

/// Numerically stable logistic sigmoid.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}

/// Solve 2×2 symmetric system `H·Δ = g`.
/// `h = [h00, h01, h11]` (upper triangle). Returns None if nearly singular.
#[inline]
pub fn solve_2x2_sym(h: [f64; 3], g: [f64; 2]) -> Option<[f64; 2]> {
    let det = h[0] * h[2] - h[1] * h[1];
    if det.abs() < 1e-14 { return None; }
    Some([(h[2] * g[0] - h[1] * g[1]) / det, (-h[1] * g[0] + h[0] * g[1]) / det])
}

/// Drive an EM loop to convergence.
///
/// `step(params)` must perform one full E→M cycle and return `(new_params, log_lik)`.
/// Convergence is declared when the relative change in log-likelihood falls below `tol`.
/// `step` is `FnMut` so the closure can update pre-allocated responsibility buffers.
pub fn run_em<P, F>(init: P, mut step: F, max_iters: u32, tol: f64) -> P
where
    F: FnMut(P) -> (P, f64),
    P: Copy,
{
    let mut params = init;
    let mut last_log_lik = f64::NEG_INFINITY;
    for _ in 0..max_iters {
        let (new_params, log_lik) = step(params);
        if last_log_lik.is_finite() {
            let scale = last_log_lik.abs().max(1.0);
            if (log_lik - last_log_lik).abs() / scale < tol {
                return new_params;
            }
        }
        last_log_lik = log_lik;
        params = new_params;
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsumexp2_numerically_stable() {
        // With inputs differing by 2000, naive exp would overflow/underflow.
        let r = logsumexp2(1000.0, -1000.0);
        assert!(r.is_finite(), "logsumexp2 overflowed");
        assert!((r - 1000.0).abs() < 1.0, "dominated by larger term");
    }

    #[test]
    fn clamp_probability_bounds() {
        assert_eq!(clamp_probability(0.0), 1e-6);
        assert_eq!(clamp_probability(1.0), 1.0 - 1e-6);
        assert_eq!(clamp_probability(0.5), 0.5);
    }

    #[test]
    fn run_em_converges() {
        // Step always returns log_lik = 0.0; second iteration sees no change → converge.
        let result = run_em(0.5f64, |p| (p, 0.0), 100, 1e-6);
        assert!(result.is_finite());
    }

    // ---- log_beta_pdf ----

    #[test]
    fn log_beta_pdf_uniform_is_zero() {
        // Beta(1, 1) is uniform on (0,1), density ≡ 1, log-density ≡ 0.
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            assert!(log_beta_pdf(x, 1.0, 1.0).abs() < 1e-12, "Beta(1,1) at {x}");
        }
    }

    #[test]
    fn log_beta_pdf_known_value_at_2_2() {
        // Beta(2, 2): f(x) = 6·x·(1-x); at x=0.5 → 1.5, log = ln(1.5).
        let got = log_beta_pdf(0.5, 2.0, 2.0);
        let want = 1.5f64.ln();
        assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
    }

    #[test]
    fn log_beta_pdf_symmetric_under_swap() {
        // f(x; α, β) = f(1-x; β, α).
        let cases = [(0.2, 3.0, 5.0), (0.7, 10.0, 2.0), (0.05, 1.5, 8.5)];
        for (x, a, b) in cases {
            let lhs = log_beta_pdf(x, a, b);
            let rhs = log_beta_pdf(1.0 - x, b, a);
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    // ---- log_normal_pdf ----

    #[test]
    fn log_normal_pdf_at_mean_is_log_normalizer() {
        // φ(μ; μ, σ) = 1/(σ·√(2π)); log = -0.5·ln(2π) - ln σ.
        let want = -0.5 * (2.0 * PI).ln() - 2.0f64.ln();
        let got = log_normal_pdf(3.0, 3.0, 2.0);
        assert!((got - want).abs() < 1e-12);
    }

    #[test]
    fn log_normal_pdf_symmetric_about_mean() {
        let mu = 1.5;
        for &d in &[0.1, 0.5, 1.2, 3.0] {
            let lhs = log_normal_pdf(mu + d, mu, 0.7);
            let rhs = log_normal_pdf(mu - d, mu, 0.7);
            assert!((lhs - rhs).abs() < 1e-12);
        }
    }

    // ---- log_poisson_pmf ----

    #[test]
    fn log_poisson_pmf_at_zero_is_neg_lambda() {
        // Pois(0; λ) = e^(-λ).
        for &lambda in &[0.5, 1.0, 3.7, 25.0] {
            let got = log_poisson_pmf(0.0, lambda);
            assert!((got - (-lambda)).abs() < 1e-12, "λ={lambda}");
        }
    }

    #[test]
    fn log_poisson_pmf_normalizes() {
        // Σ_{y=0..∞} Pois(y; λ) = 1. Truncate at λ+10·√λ which covers >>1-1e-10.
        let lambda: f64 = 4.0;
        let cap = (lambda + 10.0 * lambda.sqrt()) as u32;
        let total: f64 = (0..=cap)
            .map(|y| log_poisson_pmf(y as f64, lambda).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-9, "sum was {total}");
    }

    // ---- log_nb_pmf ----

    #[test]
    fn log_nb_pmf_collapses_to_poisson_as_theta_grows() {
        // NB(y; μ, θ) → Pois(y; μ) as θ → ∞.
        let mu = 3.0;
        let theta = 1e8;
        for &y in &[0.0, 1.0, 5.0, 10.0] {
            let nb = log_nb_pmf(y, mu, theta);
            let pois = log_poisson_pmf(y, mu);
            assert!((nb - pois).abs() < 1e-4, "y={y}: nb={nb} pois={pois}");
        }
    }

    #[test]
    fn log_nb_pmf_normalizes() {
        // Σ_y NB(y; μ, θ) = 1.
        let mu = 2.0;
        let theta = 1.5;
        // NB has fatter tails than Poisson; truncate generously.
        let total: f64 = (0..500)
            .map(|y| log_nb_pmf(y as f64, mu, theta).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-6, "sum was {total}");
    }

    // ---- log_binom_pmf ----

    #[test]
    fn log_binom_pmf_extremes() {
        // Binom(0; n, p) = (1-p)^n  →  n·ln(1-p).
        // Binom(n; n, p) = p^n      →  n·ln(p).
        let n = 10.0;
        let p = 0.3;
        let lo = log_binom_pmf(0.0, n, p);
        let hi = log_binom_pmf(n, n, p);
        assert!((lo - n * (1.0 - p).ln()).abs() < 1e-12);
        assert!((hi - n * p.ln()).abs() < 1e-12);
    }

    #[test]
    fn log_binom_pmf_normalizes() {
        let n = 20;
        let p = 0.4;
        let total: f64 = (0..=n)
            .map(|y| log_binom_pmf(y as f64, n as f64, p).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-10, "sum was {total}");
    }

    // ---- sigmoid ----

    #[test]
    fn sigmoid_at_zero_is_half() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn sigmoid_complement_and_bounded() {
        // σ(x) + σ(-x) = 1; the function stays in [0, 1] including saturated extremes.
        for &x in &[-100.0, -3.0, -0.1, 0.7, 5.0, 100.0] {
            let s = sigmoid(x);
            assert!(((s + sigmoid(-x)) - 1.0).abs() < 1e-15, "x={x}");
            assert!((0.0..=1.0).contains(&s), "x={x}: {s}");
        }
    }

    #[test]
    fn sigmoid_monotonic() {
        let xs = [-5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0];
        for w in xs.windows(2) {
            assert!(sigmoid(w[0]) < sigmoid(w[1]), "{} → {}", w[0], w[1]);
        }
    }

    // ---- solve_2x2_sym ----

    #[test]
    fn solve_2x2_sym_identity() {
        // I·Δ = g  ⇒  Δ = g.
        let d = solve_2x2_sym([1.0, 0.0, 1.0], [3.0, -4.0]).unwrap();
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] + 4.0).abs() < 1e-12);
    }

    #[test]
    fn solve_2x2_sym_known_system() {
        // H = [[2, 1],[1, 3]], g = [5, 10]  ⇒  Δ = [1, 3]
        // det = 5; Δ = (1/5) · [3·5 - 1·10, -1·5 + 2·10] = (1/5)·[5, 15] = [1, 3].
        let d = solve_2x2_sym([2.0, 1.0, 3.0], [5.0, 10.0]).unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solve_2x2_sym_singular_returns_none() {
        // [[1, 1],[1, 1]] has det 0.
        assert!(solve_2x2_sym([1.0, 1.0, 1.0], [1.0, 1.0]).is_none());
    }
}

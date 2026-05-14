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
}

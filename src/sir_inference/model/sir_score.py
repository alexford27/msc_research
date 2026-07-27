"""
Exact posterior score for the SIR Bayesian inverse problem via forward sensitivities.

Computes  grad_theta log p(theta | y)  =  grad_theta log p(y | theta) + grad_theta log pi(theta)

The likelihood gradient requires the sensitivities  ds_j = du/dtheta_j , which satisfy
their own linear ODE (the FORWARD SENSITIVITY EQUATIONS):

    d/dt (du/dtheta_j)  =  (df/du) (du/dtheta_j)  +  df/dtheta_j

These are solved AUGMENTED with the original dynamics in a single integration, giving
the trajectory and all sensitivities exactly (to solver tolerance) -- no finite differences.

This score is the only problem-specific ingredient needed by Stein thinning:
the `stein-thinning` package takes (candidates, scores) and returns the selected subset.

Companion to sir_simulator.py (same SIR RHS, same observation model).

SIR system (fractions, s + i + r = 1):
    ds/dt = -beta s i
    di/dt =  beta s i - gamma i
    dr/dt =  gamma i

Jacobians (verified symbolically):
    df/du     = [[-beta*i, -beta*s,      0],
                 [ beta*i,  beta*s-gamma, 0],
                 [      0,  gamma,        0]]
    df/dtheta = [[-s*i,  0],
                 [ s*i, -i],
                 [   0,  i]]
"""

import numpy as np
from scipy.integrate import solve_ivp

# Compartment name -> row index in the state vector
COMPARTMENT_INDEX = {"s": 0, "i": 1, "r": 2}

N_STATES = 3
N_PARAMS = 2  # (beta, gamma)


# ----------------------------------------------------------------------
# 1. THE AUGMENTED SYSTEM (states + sensitivities)
# ----------------------------------------------------------------------

def _augmented_rhs(t, z, beta, gamma):
    """
    RHS of the augmented system: original states stacked with sensitivities.

    The augmented state z has length N_STATES * (1 + N_PARAMS) = 9:
        z[0:3]  = u          = (s, i, r)
        z[3:6]  = du/dbeta   sensitivities wrt beta
        z[6:9]  = du/dgamma  sensitivities wrt gamma

    Sensitivity block j evolves as   d/dt S_j = (df/du) S_j + df/dtheta_j.
    """
    s, i, r = z[0], z[1], z[2]

    # --- original dynamics (identical to sir_simulator.sir_rhs) ---
    du = np.empty(N_STATES)
    du[0] = -beta * s * i
    du[1] = beta * s * i - gamma * i
    du[2] = gamma * i

    # --- Jacobian wrt state, df/du (3x3) ---
    J_u = np.array([
        [-beta * i, -beta * s,          0.0],
        [ beta * i,  beta * s - gamma,  0.0],
        [      0.0,  gamma,             0.0],
    ])

    # --- Jacobian wrt parameters, df/dtheta (3x2): columns are d f/d beta, d f/d gamma ---
    J_theta = np.array([
        [-s * i,  0.0],
        [ s * i,   -i],
        [   0.0,    i],
    ])

    dz = np.empty_like(z)
    dz[0:N_STATES] = du

    # Sensitivity blocks
    for j in range(N_PARAMS):
        lo = N_STATES * (1 + j)
        hi = lo + N_STATES
        S_j = z[lo:hi]
        dz[lo:hi] = J_u @ S_j + J_theta[:, j]

    return dz


def solve_sir_with_sensitivities(beta, gamma, t_eval, y0=(0.99, 0.01, 0.0),
                                 t_span=None, rtol=1e-8, atol=1e-10):
    """
    Solve the SIR system AND its parameter sensitivities in one augmented integration.

    Note on tolerances: these are tighter than the simulator's defaults because the
    score is a *derivative* -- derivative accuracy degrades faster than state accuracy,
    and a sloppy score silently corrupts Stein thinning.

    Parameters
    ----------
    beta, gamma : float          parameter values at which to evaluate
    t_eval : array, shape (T,)   times at which to report
    y0 : tuple                   initial state (s0, i0, r0), assumed KNOWN (not inferred),
                                 hence initial sensitivities are zero
    rtol, atol : float           solver tolerances

    Returns
    -------
    u : array, shape (3, T)      trajectory (s, i, r) at t_eval
    S : array, shape (2, 3, T)   S[j, c, k] = d u_c(t_k) / d theta_j,
                                 with theta = (beta, gamma)
    """
    y0 = np.asarray(y0, dtype=float)
    if not np.isclose(y0.sum(), 1.0):
        raise ValueError(f"Initial conditions must sum to 1, got {y0.sum()}")

    t_eval = np.asarray(t_eval, dtype=float)
    if t_span is None:
        t_span = (t_eval[0], t_eval[-1])

    # Augmented initial condition: states = y0, sensitivities = 0
    # (initial conditions are known constants, so du(0)/dtheta = 0)
    z0 = np.zeros(N_STATES * (1 + N_PARAMS))
    z0[0:N_STATES] = y0

    sol = solve_ivp(
        fun=_augmented_rhs,
        t_span=t_span,
        y0=z0,
        t_eval=t_eval,
        args=(beta, gamma),
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"Augmented ODE solver failed: {sol.message}")

    z = sol.y                      # shape (9, T)
    u = z[0:N_STATES]              # (3, T)
    S = np.empty((N_PARAMS, N_STATES, z.shape[1]))
    for j in range(N_PARAMS):
        lo = N_STATES * (1 + j)
        S[j] = z[lo:lo + N_STATES]

    return u, S


# ----------------------------------------------------------------------
# 2. LOG-LIKELIHOOD AND ITS GRADIENT
# ----------------------------------------------------------------------

def log_likelihood_and_grad(theta, t_obs, data, sigma,
                            y0=(0.99, 0.01, 0.0), rtol=1e-8, atol=1e-10):
    """
    Gaussian log-likelihood and its exact gradient wrt theta = (beta, gamma).

    Observation model (matching sir_simulator.observe):
        y_{c,k} = u_c(t_k; theta) + eps,   eps ~ N(0, sigma^2),  for each observed
        compartment c and observation time t_k.

    So
        log p(y|theta) = -1/(2 sigma^2) sum_c sum_k (y_{c,k} - u_c(t_k))^2 + const
    and
        d log p / d theta_j = (1/sigma^2) sum_c sum_k (y_{c,k} - u_c(t_k)) * S_j[c,k]

    Parameters
    ----------
    theta : array-like, shape (2,)   (beta, gamma)
    t_obs : array, shape (T,)        observation times
    data : dict                      {compartment_name: array shape (T,)}, e.g. from observe()
    sigma : float                    known observation noise std
    y0 : tuple                       known initial state

    Returns
    -------
    loglik : float                   log-likelihood (up to an additive constant)
    grad : array, shape (2,)         gradient wrt (beta, gamma)
    """
    beta, gamma = float(theta[0]), float(theta[1])
    t_obs = np.asarray(t_obs, dtype=float)

    u, S = solve_sir_with_sensitivities(beta, gamma, t_obs, y0=y0,
                                        rtol=rtol, atol=atol)

    loglik = 0.0
    grad = np.zeros(N_PARAMS)

    for name, y_obs in data.items():
        if name not in COMPARTMENT_INDEX:
            raise ValueError(f"Unknown compartment '{name}'")
        c = COMPARTMENT_INDEX[name]
        y_obs = np.asarray(y_obs, dtype=float)

        resid = y_obs - u[c]                       # (T,)
        loglik += -0.5 * np.sum(resid ** 2) / sigma ** 2

        for j in range(N_PARAMS):
            grad[j] += np.sum(resid * S[j, c]) / sigma ** 2

    return loglik, grad


# ----------------------------------------------------------------------
# 3. PRIORS (configurable) AND THEIR GRADIENTS
# ----------------------------------------------------------------------

class UniformBoxPrior:
    """
    Independent uniform priors on a box: beta ~ U(lo_b, hi_b), gamma ~ U(lo_g, hi_g).

    log pi(theta) is constant inside the box, so its gradient is ZERO there,
    and the density is zero (log-density -inf) outside.

    NOTE for Stein thinning: a uniform prior contributes nothing to the score inside
    the support, but the hard boundary means the Stein-class (decay) condition is not
    satisfied at the edges. If candidates sit near the boundary this can distort the
    discrepancy -- a smooth prior (e.g. log-normal) is better behaved. Flag to discuss.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def logpdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        if np.all(theta > self.lo) and np.all(theta < self.hi):
            return -np.sum(np.log(self.hi - self.lo))
        return -np.inf

    def grad_logpdf(self, theta):
        return np.zeros(N_PARAMS)


def default_prior():
    """
    The project's prior, matching init_particles() in the SMC code:
        beta  ~ U(0, 2)
        gamma ~ U(0, 1)

    Boundary caveat (checked empirically, see validate_prior_interior):
    a uniform prior has a hard cutoff, which formally violates the Stein-class
    decay condition (Stein's identity needs g(x)p(x) -> 0 at the boundary).
    In practice the likelihood decays to nothing far inside the box -- 99% of
    posterior mass lies in beta in [0.53, 0.64], gamma in [0.14, 0.22], with
    < 1e-70 of the mass within 2% of any edge -- so the boundary is never active.
    Re-run validate_prior_interior() if the data, noise level, or dimension changes.
    """
    return UniformBoxPrior(lo=[0.0, 0.0], hi=[2.0, 1.0])


class LogNormalPrior:
    """
    Independent log-normal priors: log(theta_j) ~ N(mu_j, tau_j^2), theta_j > 0.

        log pi(theta_j) = -log(theta_j) - (log theta_j - mu_j)^2 / (2 tau_j^2) + const
        d/dtheta_j log pi = -(1/theta_j) * (1 + (log theta_j - mu_j) / tau_j^2)

    Smooth and positivity-respecting -- generally the better choice for rate
    parameters, and well behaved for Stein thinning (no hard boundary).
    """

    def __init__(self, mu, tau):
        self.mu = np.asarray(mu, dtype=float)
        self.tau = np.asarray(tau, dtype=float)

    def logpdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        if np.any(theta <= 0):
            return -np.inf
        lt = np.log(theta)
        return float(np.sum(-lt - 0.5 * ((lt - self.mu) / self.tau) ** 2
                            - np.log(self.tau) - 0.5 * np.log(2 * np.pi)))

    def grad_logpdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        if np.any(theta <= 0):
            return np.full(N_PARAMS, np.nan)
        lt = np.log(theta)
        return -(1.0 / theta) * (1.0 + (lt - self.mu) / self.tau ** 2)


# ----------------------------------------------------------------------
# 4. THE POSTERIOR SCORE  (what Stein thinning consumes)
# ----------------------------------------------------------------------

def posterior_score(theta, t_obs, data, sigma, prior,
                    y0=(0.99, 0.01, 0.0), rtol=1e-8, atol=1e-10):
    """
    Exact score of the posterior at a single parameter value:

        grad_theta log p(theta | y) = grad_theta log p(y | theta) + grad_theta log pi(theta)

    The intractable normalising constant p(y) does not appear: it is independent of
    theta and so vanishes under the gradient. Only the UNNORMALISED posterior is needed.

    Returns
    -------
    score : array, shape (2,)
    """
    _, grad_ll = log_likelihood_and_grad(theta, t_obs, data, sigma,
                                         y0=y0, rtol=rtol, atol=atol)
    return grad_ll + prior.grad_logpdf(theta)


def posterior_scores(thetas, t_obs, data, sigma, prior,
                     y0=(0.99, 0.01, 0.0), rtol=1e-8, atol=1e-10,
                     n_jobs=1, verbose=False):
    """
    Scores at MANY parameter values -- the array Stein thinning needs.

    Each point requires its own augmented ODE solve, and the solves are completely
    independent, so this is embarrassingly parallel (set n_jobs > 1, needs joblib).

    Parameters
    ----------
    thetas : array, shape (M, 2)   candidate parameter values
    n_jobs : int                   parallel workers (1 = serial; -1 = all cores)

    Returns
    -------
    scores : array, shape (M, 2)   scores[m] = grad log p(thetas[m] | y)
    """
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    M = thetas.shape[0]

    def _one(m):
        return posterior_score(thetas[m], t_obs, data, sigma, prior,
                               y0=y0, rtol=rtol, atol=atol)

    if n_jobs == 1:
        scores = np.empty((M, N_PARAMS))
        for m in range(M):
            if verbose and m % max(1, M // 10) == 0:
                print(f"  score {m}/{M}")
            scores[m] = _one(m)
        return scores

    from joblib import Parallel, delayed
    out = Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_one)(m) for m in range(M)
    )
    return np.array(out)


# ----------------------------------------------------------------------
# 5. VALIDATION: analytic score vs finite differences
#    (the non-negotiable correctness check)
# ----------------------------------------------------------------------

def finite_difference_score(theta, t_obs, data, sigma, prior,
                            y0=(0.99, 0.01, 0.0), eps=1e-6):
    """
    Central-difference approximation of the posterior score, for VALIDATION ONLY.

    Deliberately uses the plain (non-augmented) solve so it is an independent check
    on the sensitivity machinery, not a re-use of it.
    """
    theta = np.asarray(theta, dtype=float)

    def log_post(th):
        beta, gamma = float(th[0]), float(th[1])
        u, _ = solve_sir_with_sensitivities(beta, gamma, t_obs, y0=y0)
        ll = 0.0
        for name, y_obs in data.items():
            c = COMPARTMENT_INDEX[name]
            resid = np.asarray(y_obs, dtype=float) - u[c]
            ll += -0.5 * np.sum(resid ** 2) / sigma ** 2
        return ll + prior.logpdf(th)

    grad = np.zeros(N_PARAMS)
    for j in range(N_PARAMS):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        grad[j] = (log_post(tp) - log_post(tm)) / (2 * eps)
    return grad


def validate_prior_interior(t_obs, data, sigma, prior,
                            y0=(0.99, 0.01, 0.0), n_grid=120,
                            edge_frac=0.02, verbose=True):
    """
    Check that posterior mass sits in the INTERIOR of a uniform prior box.

    Why this matters: a uniform prior has a hard cutoff, which formally violates the
    Stein-class decay condition underlying Stein's identity (which needs g(x)p(x) -> 0
    at the boundary). If the posterior is concentrated well inside the box, the
    likelihood does the decaying and the boundary is never active, so KSD/Stein
    thinning behave correctly. If mass presses against an edge, the discrepancy can
    misbehave -- widen the box or switch to a smooth prior.

    Worth re-running whenever the data, noise level, observation model, or dimension
    changes (e.g. the R0 ridge could reach an edge in a different configuration).

    Returns
    -------
    safe : bool                 True if boundary mass is negligible
    edge_mass : float           posterior mass within edge_frac of any box edge
    hpd_box : dict              99% mass interval per parameter
    """
    if not isinstance(prior, UniformBoxPrior):
        raise TypeError("This check applies to UniformBoxPrior only.")

    lo, hi = prior.lo, prior.hi
    eps = 1e-3
    grids = [np.linspace(lo[j] + eps, hi[j] - eps, n_grid) for j in range(N_PARAMS)]

    logpost = np.full((n_grid, n_grid), -np.inf)  # [gamma_idx, beta_idx]
    for a, g in enumerate(grids[1]):
        for b, be in enumerate(grids[0]):
            try:
                u, _ = solve_sir_with_sensitivities(be, g, t_obs, y0=y0,
                                                    rtol=1e-6, atol=1e-8)
                ll = 0.0
                for name, y_obs in data.items():
                    c = COMPARTMENT_INDEX[name]
                    ll += -0.5 * np.sum((np.asarray(y_obs) - u[c]) ** 2) / sigma ** 2
                logpost[a, b] = ll
            except Exception:
                pass

    post = np.exp(logpost - np.nanmax(logpost))
    post /= np.nansum(post)

    hpd_box = {}
    for j, name in enumerate(("beta", "gamma")):
        marg = post.sum(axis=0) if j == 0 else post.sum(axis=1)
        c = np.cumsum(marg)
        hpd_box[name] = (float(grids[j][np.searchsorted(c, 0.005)]),
                         float(grids[j][np.searchsorted(c, 0.995)]))

    width = hi - lo
    edge_mass = 0.0
    for a, g in enumerate(grids[1]):
        for b, be in enumerate(grids[0]):
            if (be < lo[0] + edge_frac * width[0] or be > hi[0] - edge_frac * width[0]
                    or g < lo[1] + edge_frac * width[1] or g > hi[1] - edge_frac * width[1]):
                edge_mass += post[a, b]

    safe = edge_mass < 1e-6

    if verbose:
        print(f"  prior box     : beta {tuple(lo[0:1]) + tuple(hi[0:1])}, "
              f"gamma {tuple(lo[1:2]) + tuple(hi[1:2])}")
        print(f"  99% mass      : beta {hpd_box['beta']}, gamma {hpd_box['gamma']}")
        print(f"  mass within {edge_frac:.0%} of an edge: {edge_mass:.2e}"
              f"   -> {'SAFE (boundary never active)' if safe else 'WARNING: mass near boundary'}")

    return safe, float(edge_mass), hpd_box


def validate_score(theta, t_obs, data, sigma, prior,
                   y0=(0.99, 0.01, 0.0), eps=1e-6, tol=1e-4, verbose=True):
    """
    Check the analytic (forward-sensitivity) score against finite differences.

    Returns
    -------
    passed : bool
    analytic, numeric, rel_err : arrays
    """
    analytic = posterior_score(theta, t_obs, data, sigma, prior, y0=y0)
    numeric = finite_difference_score(theta, t_obs, data, sigma, prior,
                                      y0=y0, eps=eps)
    denom = np.maximum(np.abs(numeric), 1e-12)
    rel_err = np.abs(analytic - numeric) / denom
    passed = bool(np.all(rel_err < tol))

    if verbose:
        print(f"  theta = ({theta[0]:.4f}, {theta[1]:.4f})")
        print(f"    analytic : {analytic}")
        print(f"    numeric  : {numeric}")
        print(f"    rel err  : {rel_err}   -> {'PASS' if passed else 'FAIL'}")

    return passed, analytic, numeric, rel_err


# ----------------------------------------------------------------------
# DEMO / VALIDATION SUITE
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Ground truth settings (match sir_simulator demo) ---
    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    # --- Generate a dataset (full observation) ---
    u_true, _ = solve_sir_with_sensitivities(BETA_TRUE, GAMMA_TRUE, T_OBS, y0=Y0)
    data = {c: u_true[COMPARTMENT_INDEX[c]] + rng.normal(0, SIGMA, T_OBS.size)
            for c in ("s", "i", "r")}

    # --- Prior: log-normal centred near the truth (smooth, positivity-respecting) ---
    prior = LogNormalPrior(mu=np.log([0.5, 0.25]), tau=[0.5, 0.5])

    print("=" * 70)
    print("VALIDATION: analytic forward-sensitivity score vs finite differences")
    print("=" * 70)

    test_points = [
        (BETA_TRUE, GAMMA_TRUE),   # at the truth
        (0.45, 0.15),              # below
        (0.80, 0.30),              # above
        (0.30, 0.30),              # low R0
        (0.90, 0.15),              # high R0
    ]

    all_passed = True
    for th in test_points:
        passed, *_ = validate_score(np.array(th), T_OBS, data, SIGMA, prior, y0=Y0)
        all_passed &= passed
        print()

    print("=" * 70)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)

    # --- Demo: scores at a batch of candidates (the Stein-thinning input) ---
    print("\nBatch score computation (the array stein-thinning consumes):")
    cand = np.column_stack([
        rng.uniform(0.4, 0.8, 20),
        rng.uniform(0.1, 0.3, 20),
    ])
    sc = posterior_scores(cand, T_OBS, data, SIGMA, prior, y0=Y0)
    print(f"  candidates shape : {cand.shape}")
    print(f"  scores shape     : {sc.shape}")
    print(f"  first 3 scores   :\n{sc[:3]}")
    print("\n  -> pass (cand, sc) to stein_thinning.thinning.thin(...)")
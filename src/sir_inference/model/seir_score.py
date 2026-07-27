"""
SEIR model and exact posterior score via forward sensitivities -- the 5D tier.

Extends the SIR machinery (sir_score.py) to a five-parameter problem:

    theta = (beta, sigma_E, gamma, e0, i0)

SEIR system (fractions, s + e + i + r = 1):
    ds/dt = -beta s i
    de/dt =  beta s i - sigma_E e
    di/dt =  sigma_E e - gamma i
    dr/dt =  gamma i

with sigma_E the incubation rate (1/sigma_E = mean latent period).

WHY THESE FIVE PARAMETERS
-------------------------
SEIR itself contributes only three rate parameters. The dimensional ladder is
completed with the two unknown initial fractions e0 and i0, which is both
realistic (the initial state of an epidemic is rarely known) and useful for the
study: initial conditions are weakly identified from trajectory data, so they
stress identifiability in a way extra rate parameters would not. r0 is fixed at
zero and s0 = 1 - e0 - i0 by conservation.

THE STRUCTURAL DIFFERENCE FROM SIR
----------------------------------
In sir_score.py the initial conditions were known constants, so every sensitivity
started at zero. Here two of the parameters ARE initial conditions, and their
sensitivity blocks start non-zero. Because s0 absorbs the conservation
constraint, perturbing e0 moves mass OUT of s and INTO e:

    d u0 / d e0 = (-1, 1, 0, 0),      d u0 / d i0 = (-1, 0, 1, 0)

Using unit vectors here instead would be a silent error -- the score would be
wrong only in the components that matter most for the weakly-identified
parameters. The finite-difference check at the bottom of this file is what
catches it.

Jacobians (verified symbolically):
    df/du = [[-beta i,      0, -beta s, 0],
             [ beta i, -sig_E,  beta s, 0],
             [      0,  sig_E,  -gamma, 0],
             [      0,      0,   gamma, 0]]

    df/dtheta (rate parameters only; columns beta, sigma_E, gamma) =
            [[-s i,  0,  0],
             [ s i, -e,  0],
             [   0,  e, -i],
             [   0,  0,  i]]

    The e0 and i0 columns of df/dtheta are ZERO -- initial conditions do not
    appear in the vector field, they enter only through the initial condition of
    the sensitivity system.
"""

import numpy as np
from scipy.integrate import solve_ivp

COMPARTMENT_INDEX = {"s": 0, "e": 1, "i": 2, "r": 3}
PARAM_NAMES = ["beta", "sigma_E", "gamma", "e0", "i0"]

N_STATES = 4
N_PARAMS = 5
N_RATE_PARAMS = 3          # beta, sigma_E, gamma appear in the vector field
IC_PARAM_INDEX = {3: "e", 4: "i"}   # param index -> compartment it initialises


# ----------------------------------------------------------------------
# 1. THE MODEL
# ----------------------------------------------------------------------

def seir_rhs(t, u, beta, sigma_E, gamma):
    """Right-hand side of the SEIR system in fractions."""
    s, e, i, r = u
    return np.array([
        -beta * s * i,
        beta * s * i - sigma_E * e,
        sigma_E * e - gamma * i,
        gamma * i,
    ])


def initial_state(e0, i0, r0=0.0):
    """Initial state with s0 absorbing the conservation constraint."""
    return np.array([1.0 - e0 - i0 - r0, e0, i0, r0])


def solve_seir(theta, t_eval, r0=0.0, rtol=1e-8, atol=1e-10):
    """Solve SEIR (states only) at theta = (beta, sigma_E, gamma, e0, i0)."""
    beta, sigma_E, gamma, e0, i0 = map(float, theta)
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(seir_rhs, (t_eval[0], t_eval[-1]), initial_state(e0, i0, r0),
                    t_eval=t_eval, args=(beta, sigma_E, gamma),
                    method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"SEIR solver failed: {sol.message}")
    return sol.y


# ----------------------------------------------------------------------
# 2. THE AUGMENTED SYSTEM (states + sensitivities)
# ----------------------------------------------------------------------

def _augmented_rhs(t, z, beta, sigma_E, gamma):
    """
    Augmented state of length N_STATES * (1 + N_PARAMS) = 24:
        z[0:4]   = u
        z[4:8]   = du/dbeta
        z[8:12]  = du/dsigma_E
        z[12:16] = du/dgamma
        z[16:20] = du/de0
        z[20:24] = du/di0

    Every sensitivity block obeys  d/dt S_j = (df/du) S_j + df/dtheta_j.
    For the initial-condition parameters df/dtheta_j = 0, so those blocks are
    driven purely by the state Jacobian -- they evolve from a non-zero initial
    condition rather than being forced.
    """
    s, e, i, r = z[0], z[1], z[2], z[3]

    du = np.array([
        -beta * s * i,
        beta * s * i - sigma_E * e,
        sigma_E * e - gamma * i,
        gamma * i,
    ])

    J_u = np.array([
        [-beta * i,      0.0, -beta * s, 0.0],
        [ beta * i, -sigma_E,  beta * s, 0.0],
        [      0.0,  sigma_E,    -gamma, 0.0],
        [      0.0,      0.0,     gamma, 0.0],
    ])

    # Columns for the RATE parameters only; IC parameters contribute zero here.
    J_theta = np.array([
        [-s * i,  0.0,  0.0],
        [ s * i,   -e,  0.0],
        [   0.0,    e,   -i],
        [   0.0,  0.0,    i],
    ])

    dz = np.empty_like(z)
    dz[0:N_STATES] = du
    for j in range(N_PARAMS):
        lo = N_STATES * (1 + j)
        S_j = z[lo:lo + N_STATES]
        forcing = J_theta[:, j] if j < N_RATE_PARAMS else 0.0
        dz[lo:lo + N_STATES] = J_u @ S_j + forcing
    return dz


def _initial_sensitivities(r0=0.0):
    """
    Initial condition for the sensitivity system, shape (N_PARAMS, N_STATES).

    Rate parameters: the initial state does not depend on them -> zero.
    Initial-condition parameters: s0 = 1 - e0 - i0 - r0 absorbs the constraint,
    so raising e0 by one unit lowers s0 by one unit:
        d u0 / d e0 = (-1, 1, 0, 0),  d u0 / d i0 = (-1, 0, 1, 0).
    """
    S0 = np.zeros((N_PARAMS, N_STATES))
    for j, comp in IC_PARAM_INDEX.items():
        S0[j, COMPARTMENT_INDEX["s"]] = -1.0
        S0[j, COMPARTMENT_INDEX[comp]] = 1.0
    return S0


def solve_seir_with_sensitivities(theta, t_eval, r0=0.0, rtol=1e-8, atol=1e-10):
    """
    Solve SEIR and all five parameter sensitivities in one augmented integration.

    Returns
    -------
    u : (4, T)          trajectory
    S : (5, 4, T)       S[j, c, k] = d u_c(t_k) / d theta_j
    """
    beta, sigma_E, gamma, e0, i0 = map(float, theta)
    t_eval = np.asarray(t_eval, dtype=float)

    z0 = np.zeros(N_STATES * (1 + N_PARAMS))
    z0[0:N_STATES] = initial_state(e0, i0, r0)
    S0 = _initial_sensitivities(r0)
    for j in range(N_PARAMS):
        lo = N_STATES * (1 + j)
        z0[lo:lo + N_STATES] = S0[j]

    sol = solve_ivp(_augmented_rhs, (t_eval[0], t_eval[-1]), z0, t_eval=t_eval,
                    args=(beta, sigma_E, gamma), method="RK45",
                    rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"Augmented SEIR solver failed: {sol.message}")

    z = sol.y
    u = z[0:N_STATES]
    S = np.empty((N_PARAMS, N_STATES, z.shape[1]))
    for j in range(N_PARAMS):
        lo = N_STATES * (1 + j)
        S[j] = z[lo:lo + N_STATES]
    return u, S


# ----------------------------------------------------------------------
# 3. LIKELIHOOD, PRIOR, SCORE
# ----------------------------------------------------------------------

def log_likelihood_and_grad(theta, t_obs, data, sigma, r0=0.0,
                            rtol=1e-8, atol=1e-10):
    """Gaussian log-likelihood and its exact gradient (5 components)."""
    u, S = solve_seir_with_sensitivities(theta, t_obs, r0=r0, rtol=rtol, atol=atol)
    loglik = 0.0
    grad = np.zeros(N_PARAMS)
    for name, y_obs in data.items():
        c = COMPARTMENT_INDEX[name]
        resid = np.asarray(y_obs, dtype=float) - u[c]
        loglik += -0.5 * np.sum(resid ** 2) / sigma ** 2
        for j in range(N_PARAMS):
            grad[j] += np.sum(resid * S[j, c]) / sigma ** 2
    return loglik, grad


class UniformBoxPrior5D:
    """
    Independent uniform priors on a box over (beta, sigma_E, gamma, e0, i0).

    Same boundary caveat as the 2D case: the hard cutoff formally violates the
    Stein-class decay condition. This matters MORE here -- e0 and i0 are weakly
    identified and bounded below by zero, so posterior mass may genuinely sit near
    an edge. Check with the interior diagnostic before trusting KSD-based results.
    """

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def logpdf(self, theta):
        theta = np.asarray(theta, dtype=float)
        if np.all(theta > self.lo) and np.all(theta < self.hi):
            return -float(np.sum(np.log(self.hi - self.lo)))
        return -np.inf

    def grad_logpdf(self, theta):
        return np.zeros(N_PARAMS)


def default_prior():
    """
    Priors for the 5D tier. Rate ranges mirror the 2D study; the initial
    fractions are given generous support around plausible outbreak seeds.
    """
    return UniformBoxPrior5D(lo=[0.0, 0.0, 0.0, 0.0, 0.0],
                             hi=[2.0, 1.0, 1.0, 0.10, 0.10])


def posterior_score(theta, t_obs, data, sigma, prior, r0=0.0,
                    rtol=1e-8, atol=1e-10):
    """Exact score of the SEIR posterior at one parameter value."""
    _, grad_ll = log_likelihood_and_grad(theta, t_obs, data, sigma, r0=r0,
                                         rtol=rtol, atol=atol)
    return grad_ll + prior.grad_logpdf(theta)


def posterior_scores(thetas, t_obs, data, sigma, prior, r0=0.0,
                     rtol=1e-8, atol=1e-10, n_jobs=1, verbose=False):
    """Scores at many parameter values; embarrassingly parallel over points."""
    thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
    M = thetas.shape[0]

    def _one(m):
        return posterior_score(thetas[m], t_obs, data, sigma, prior,
                               r0=r0, rtol=rtol, atol=atol)

    if n_jobs == 1:
        out = np.empty((M, N_PARAMS))
        for m in range(M):
            out[m] = _one(m)
        return out

    from joblib import Parallel, delayed
    return np.array(Parallel(n_jobs=n_jobs, verbose=5 if verbose else 0)(
        delayed(_one)(m) for m in range(M)))


# ----------------------------------------------------------------------
# 4. DATA GENERATION
# ----------------------------------------------------------------------

def simulate_dataset(theta_true, t_obs, sigma, observed=("i",), r0=0.0, seed=None):
    """Generate a noisy SEIR dataset. Returns (truth, data dict)."""
    rng = np.random.default_rng(seed)
    truth = solve_seir(theta_true, t_obs, r0=r0)
    data = {c: truth[COMPARTMENT_INDEX[c]] + rng.normal(0, sigma, len(t_obs))
            for c in observed}
    return truth, data


# ----------------------------------------------------------------------
# 5. VALIDATION
# ----------------------------------------------------------------------

def finite_difference_score(theta, t_obs, data, sigma, prior, r0=0.0, eps=1e-6):
    """Central-difference score, for validation only."""
    theta = np.asarray(theta, dtype=float)

    def log_post(th):
        u = solve_seir(th, t_obs, r0=r0)
        ll = 0.0
        for name, y_obs in data.items():
            c = COMPARTMENT_INDEX[name]
            ll += -0.5 * np.sum((np.asarray(y_obs) - u[c]) ** 2) / sigma ** 2
        return ll + prior.logpdf(th)

    g = np.zeros(N_PARAMS)
    for j in range(N_PARAMS):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        g[j] = (log_post(tp) - log_post(tm)) / (2 * eps)
    return g


def validate_score(theta, t_obs, data, sigma, prior, r0=0.0, eps=1e-6,
                   tol=1e-4, verbose=True):
    """Check the analytic score against finite differences, component by component."""
    a = posterior_score(theta, t_obs, data, sigma, prior, r0=r0)
    n = finite_difference_score(theta, t_obs, data, sigma, prior, r0=r0, eps=eps)
    rel = np.abs(a - n) / np.maximum(np.abs(n), 1e-10)
    ok = bool(np.all(rel < tol))
    if verbose:
        print(f"  theta = {np.round(theta, 4)}")
        for j, name in enumerate(PARAM_NAMES):
            flag = "ok" if rel[j] < tol else "FAIL"
            print(f"    {name:9s} analytic {a[j]:14.5f}  numeric {n[j]:14.5f}  "
                  f"rel {rel[j]:9.2e}  {flag}")
        print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok, a, n, rel


if __name__ == "__main__":
    # Truth: R0 = beta/gamma = 3, mean latent period 1/sigma_E = 5 days
    THETA_TRUE = np.array([0.6, 0.2, 0.2, 0.005, 0.005])
    SIGMA = 0.05
    T_OBS = np.linspace(0, 80, 40)

    truth, data = simulate_dataset(THETA_TRUE, T_OBS, SIGMA,
                                   observed=("i",), seed=2327)
    prior = default_prior()

    print("=" * 74)
    print("SEIR SCORE VALIDATION (5 parameters, infectious-only observation)")
    print("=" * 74)
    print(f"truth: " + ", ".join(f"{n}={v}" for n, v in zip(PARAM_NAMES, THETA_TRUE)))
    print(f"R0 = {THETA_TRUE[0]/THETA_TRUE[2]:.2f}, "
          f"latent period = {1/THETA_TRUE[1]:.1f} days\n")

    tests = [
        THETA_TRUE,
        np.array([0.50, 0.25, 0.18, 0.004, 0.006]),
        np.array([0.75, 0.15, 0.25, 0.008, 0.003]),
        np.array([0.60, 0.20, 0.20, 0.010, 0.010]),
    ]
    all_ok = True
    for th in tests:
        ok, *_ = validate_score(th, T_OBS, data, SIGMA, prior)
        all_ok &= ok
        print()

    print("=" * 74)
    print(f"OVERALL: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 74)
    print("\nNote: the e0 and i0 components are the ones that would fail if the")
    print("initial sensitivity conditions were set to unit vectors rather than")
    print("accounting for s0 absorbing the conservation constraint.")
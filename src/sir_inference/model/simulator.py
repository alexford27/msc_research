"""
SIR simulator for Bayesian inverse problem (MSc dissertation).

Design principles (per project plan):
  - Work in FRACTIONS: s + i + r = 1  (frequency-dependent transmission).
  - Base case infers (beta, gamma) only; sigma, initial conditions, N are fixed/known.
  - Three steps are cleanly SEPARATED and reusable:
        1. solve_sir(...)      -> the deterministic forward model (the "expensive" likelihood ingredient)
        2. generate_data(...)  -> sample the true trajectory at observation times
        3. observe(...)        -> add measurement noise to chosen compartments (SWAPPABLE observation model)
    This separation lets you reuse solve_sir inside the SMC/GP-HM likelihood,
    and lets you switch the observation model (full vs infectious-only) by a flag.

Requires: numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.integrate import solve_ivp


# ----------------------------------------------------------------------
# 1. THE FORWARD MODEL (deterministic ODE solve)
#    This is the object the likelihood calls once per parameter evaluation.
# ----------------------------------------------------------------------

def sir_rhs(t, y, beta, gamma):
    """
    Right-hand side of the SIR ODEs in FRACTIONS (s + i + r = 1).

        ds/dt = -beta * s * i
        di/dt =  beta * s * i - gamma * i
        dr/dt =  gamma * i

    Parameters
    ----------
    t : float                 time (unused explicitly; ODE is autonomous)
    y : array-like, shape (3,) current state [s, i, r]
    beta : float              transmission rate
    gamma : float             recovery rate (1/gamma = mean infectious period)
    """
    s, i, r = y
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return [ds, di, dr]


def solve_sir(beta, gamma, t_eval, y0=(0.99, 0.01, 0.0),
              t_span=None, rtol=1e-8, atol=1e-10):
    """
    Solve the SIR system forward and return the trajectory at t_eval.

    Parameters
    ----------
    beta, gamma : float        parameters
    t_eval : array, shape (T,) times at which to report the solution
    y0 : tuple (s0, i0, r0)    initial fractions (must sum to 1)
    t_span : (t0, t1) or None  integration interval; defaults to [t_eval[0], t_eval[-1]]
    rtol, atol : float         solver tolerances (tight, so solver error << noise)

    Returns
    -------
    sol_y : array, shape (3, T)  rows are s(t), i(t), r(t) at the t_eval points
    """
    y0 = np.asarray(y0, dtype=float)
    # Sanity: initial fractions sum to 1
    if not np.isclose(y0.sum(), 1.0):
        raise ValueError(f"Initial conditions must sum to 1, got {y0.sum()}")

    t_eval = np.asarray(t_eval, dtype=float)
    if t_span is None:
        t_span = (t_eval[0], t_eval[-1])

    sol = solve_ivp(
        fun=sir_rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(beta, gamma),
        method="RK45",        # good default; switch to "LSODA"/"Radau" if stiff
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # sol.y has shape (3, T): rows s, i, r
    return sol.y


# ----------------------------------------------------------------------
# 2. GENERATE THE TRUE (NOISE-FREE) DATA AT OBSERVATION TIMES
# ----------------------------------------------------------------------

def generate_truth(beta_true, gamma_true, t_obs, y0=(0.99, 0.01, 0.0)):
    """
    Produce the noise-free trajectory at the observation times.

    Returns
    -------
    truth : array, shape (3, T)  true [s, i, r] at t_obs (the ground truth)
    """
    return solve_sir(beta_true, gamma_true, t_eval=t_obs, y0=y0)


# ----------------------------------------------------------------------
# 3. OBSERVATION MODEL (add noise to chosen compartments) -- SWAPPABLE
# ----------------------------------------------------------------------

def observe(truth, sigma, observed=("s", "i", "r"), rng=None):
    """
    Apply the measurement model: additive Gaussian noise on chosen compartments.

    This is the piece you SWITCH for different experiments:
      - observed = ("s", "i", "r")  -> full observation (easiest; for validation)
      - observed = ("i",)           -> realistic infectious-only observation (harder)
      - observed = ("s", "i")       -> two compartments (r determined by conservation)

    Parameters
    ----------
    truth : array, shape (3, T)   noise-free [s, i, r]
    sigma : float                 measurement-noise standard deviation (fixed/known)
    observed : tuple of str       which compartments are observed
    rng : np.random.Generator     for reproducibility (pass np.random.default_rng(seed))

    Returns
    -------
    data : dict  {compartment_name: noisy_observations array shape (T,)}
    """
    if rng is None:
        rng = np.random.default_rng()

    index = {"s": 0, "i": 1, "r": 2}
    data = {}
    for name in observed:
        if name not in index:
            raise ValueError(f"Unknown compartment '{name}'")
        clean = truth[index[name]]
        noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
        data[name] = noisy
    return data


# ----------------------------------------------------------------------
# CONVENIENCE: full pipeline in one call
# ----------------------------------------------------------------------

def simulate_dataset(beta_true, gamma_true, t_obs, sigma,
                     y0=(0.99, 0.01, 0.0),
                     observed=("s", "i", "r"), seed=None):
    """
    One-shot: solve -> generate truth -> observe.

    Returns
    -------
    t_obs : array (T,)
    truth : array (3, T)            ground-truth trajectory (for plotting/checking)
    data  : dict                    noisy observations for the observed compartments
    meta  : dict                    the settings used (handy for logging / captions)
    """
    rng = np.random.default_rng(seed)
    truth = generate_truth(beta_true, gamma_true, t_obs, y0=y0)
    data = observe(truth, sigma, observed=observed, rng=rng)
    meta = dict(beta_true=beta_true, gamma_true=gamma_true, sigma=sigma,
                y0=y0, observed=observed, seed=seed,
                R0=beta_true / gamma_true)
    return t_obs, truth, data, meta


# ----------------------------------------------------------------------
# DEMO / SANITY CHECKS  (run this file directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Ground-truth settings (the values your inference will try to recover) ---
    BETA_TRUE = 0.6      # transmission rate
    GAMMA_TRUE = 0.2     # recovery rate  -> R0 = beta/gamma = 3.0
    SIGMA = 0.02         # measurement noise std (fixed/known for the base case)
    T_OBS = np.linspace(0, 60, 30)   # 30 evenly spaced observation times over 60 days
    Y0 = (0.99, 0.01, 0.0)           # 1% initially infectious

    # --- Run the full pipeline (full observation for validation) ---
    t_obs, truth, data, meta = simulate_dataset(
        BETA_TRUE, GAMMA_TRUE, T_OBS, SIGMA,
        y0=Y0, observed=("s", "i", "r"), seed=0,
    )

    print("Settings:", meta)

    # --- SANITY CHECK 1: compartments sum to 1 (conservation law) ---
    sums = truth.sum(axis=0)
    print(f"Conservation check: s+i+r ranges in "
          f"[{sums.min():.6f}, {sums.max():.6f}] (should be ~1.0)")

    # --- SANITY CHECK 2: fractions stay in [0, 1] ---
    print(f"s in [{truth[0].min():.3f}, {truth[0].max():.3f}], "
          f"i in [{truth[1].min():.3f}, {truth[1].max():.3f}], "
          f"r in [{truth[2].min():.3f}, {truth[2].max():.3f}]")

    # --- PLOT: true trajectories + noisy observations ---
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = {"s": "Susceptible", "i": "Infectious", "r": "Recovered"}
    colors = {"s": "tab:blue", "i": "tab:red", "r": "tab:green"}
    idx = {"s": 0, "i": 1, "r": 2}

    # Smooth true curves on a fine grid (for display only)
    t_fine = np.linspace(T_OBS[0], T_OBS[-1], 400)
    truth_fine = solve_sir(BETA_TRUE, GAMMA_TRUE, t_eval=t_fine, y0=Y0)
    for c in ("s", "i", "r"):
        ax.plot(t_fine, truth_fine[idx[c]], color=colors[c], lw=2, label=labels[c])
    # Noisy observations as points
    for c in data:
        ax.scatter(t_obs, data[c], color=colors[c], s=18, alpha=0.7,
                   edgecolor="k", linewidth=0.3)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Population fraction")
    ax.set_title(f"SIR simulation  (beta={BETA_TRUE}, gamma={GAMMA_TRUE}, "
                 f"R0={meta['R0']:.1f}, sigma={SIGMA})")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    plt.show()

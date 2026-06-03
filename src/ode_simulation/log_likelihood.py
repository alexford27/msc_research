"""
Gaussian log-likelihood for the SIR Bayesian inverse problem.

This is the SHARED ingredient that both RWMH and SMC will call. It wraps the
forward model (solve_sir) and scores how well a candidate (beta, gamma) explains
the observed data, under the SAME noise model used to generate the data in observe():
    - additive Gaussian noise
    - independent across time points AND across compartments
    - standard deviation sigma (KNOWN / fixed in the base case)

Key design choices (see notes in the dissertation methodology):
  - Works in LOG space (products of densities -> sums; avoids underflow; samplers
    use log-likelihoods anyway).
  - Sums only over the OBSERVED compartments (whatever keys are in `data`), so it
    automatically adapts to full vs partial (e.g. infectious-only) observation.
  - sigma is an ARGUMENT (known), not inferred, in the base case.
  - Returns -inf if the ODE solve fails (an un-solvable parameter is infinitely
    implausible -> the sampler simply rejects it), rather than crashing.

Depends on solve_sir from the simulator module.
"""

import numpy as np
from ode_simulator import solve_sir   # adjust import path to your project layout


# ----------------------------------------------------------------------
# THE LOG-LIKELIHOOD
# ----------------------------------------------------------------------

def log_likelihood(beta, gamma, t_obs, data, sigma,
                   y0=(0.99, 0.01, 0.0)):
    """
    Gaussian log-likelihood of `data` under SIR parameters (beta, gamma).

    Parameters
    ----------
    beta, gamma : float        candidate parameters to score
    t_obs : array, shape (T,)  observation times (must match how `data` was made)
    data : dict                {compartment_name: noisy observations (T,)}
                               e.g. {"s": ..., "i": ..., "r": ...} or {"i": ...}
    sigma : float              measurement-noise std (KNOWN; must match observe())
    y0 : tuple (s0, i0, r0)    initial fractions (fixed/known)

    Returns
    -------
    loglik : float             the log-likelihood; -inf if the solve fails or
                               parameters are invalid
    """
    # --- Guard against invalid parameters a sampler might propose ---
    # Rates must be positive; a non-positive rate is outside the model's support.
    if beta <= 0 or gamma <= 0:
        return -np.inf

    # --- 1. Forward model: solve the ODE at this (beta, gamma) ---
    # This is the expensive step (one ODE solve = one likelihood evaluation).
    try:
        trajectory = solve_sir(beta, gamma, t_eval=t_obs, y0=y0)  # shape (3, T)
    except Exception:
        # Solver failed (e.g. extreme parameters) -> infinitely implausible.
        return -np.inf

    # --- 2. Compare prediction to data under the Gaussian noise model ---
    index = {"s": 0, "i": 1, "r": 2}

    # Accumulate the sum of squared residuals over all observed compartments
    # and count the total number of scalar observations.
    sum_sq_resid = 0.0
    n_obs = 0
    for name, observed_values in data.items():
        predicted = trajectory[index[name]]            # model prediction for this compartment
        residuals = observed_values - predicted        # data minus model
        sum_sq_resid += np.sum(residuals ** 2)
        n_obs += observed_values.size

    # --- 3. Assemble the Gaussian log-likelihood ---
    #   log N(y; x, sigma^2) summed over all observations:
    #   = -(1/(2 sigma^2)) * sum_sq_resid  -  (n_obs/2) * log(2 pi sigma^2)
    loglik = (-0.5 / sigma**2) * sum_sq_resid \
             - 0.5 * n_obs * np.log(2.0 * np.pi * sigma**2)

    return loglik


# ----------------------------------------------------------------------
# DEMO / SANITY CHECKS  (run this file directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from ode_simulator import simulate_dataset

    # --- Generate a dataset with KNOWN ground truth ---
    BETA_TRUE = 0.6
    GAMMA_TRUE = 0.2
    SIGMA = 0.1
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    t_obs, truth, data, meta = simulate_dataset(
        BETA_TRUE, GAMMA_TRUE, T_OBS, SIGMA,
        y0=Y0, observed=("s", "i", "r"), seed=0,
    )

    # --- SANITY CHECK 1: truth should score NEAR THE TOP, far above wrong params ---
    # NOTE: because the data is noisy, the exact maximum-likelihood point sits at a
    # nearby parameter, NOT exactly at the truth. What we expect is that the truth
    # scores far higher than clearly-wrong parameters; it need not beat every nearby
    # point. (This is exactly why we want the POSTERIOR, which quantifies uncertainty
    # and should COVER the truth, rather than just a single best-fit point.)
    ll_true = log_likelihood(BETA_TRUE, GAMMA_TRUE, t_obs, data, SIGMA, y0=Y0)
    ll_wrong = log_likelihood(0.9, 0.4, t_obs, data, SIGMA, y0=Y0)  # wrong params
    ll_very_wrong = log_likelihood(0.3, 0.5, t_obs, data, SIGMA, y0=Y0)

    print(f"log-likelihood at TRUE  (0.6, 0.2): {ll_true:10.2f}")
    print(f"log-likelihood at WRONG (0.9, 0.4): {ll_wrong:10.2f}")
    print(f"log-likelihood at V.WRONG(0.3,0.5): {ll_very_wrong:10.2f}")
    print("--> truth should score NEAR THE TOP, far above clearly-wrong params;")
    print("    noise shifts the exact maximum to a nearby point, not exactly the truth.")

    # --- SANITY CHECK 2: invalid params return -inf, not a crash ---
    ll_negative = log_likelihood(-0.1, 0.2, t_obs, data, SIGMA, y0=Y0)
    print(f"\nlog-likelihood at INVALID (-0.1, 0.2): {ll_negative} (should be -inf)")

    # --- SANITY CHECK 3: a quick 1D slice along beta (gamma fixed at truth) ---
    # The maximum should sit near beta_true = 0.6.
    betas = np.linspace(0.3, 1.0, 15)
    print("\nbeta   log-likelihood (gamma fixed at 0.2):")
    for b in betas:
        ll = log_likelihood(b, GAMMA_TRUE, t_obs, data, SIGMA, y0=Y0)
        marker = "  <-- truth" if abs(b - BETA_TRUE) < 0.03 else ""
        print(f"{b:4.2f}   {ll:10.2f}{marker}")

    # --- VISUALISE THE RIDGE: 2D log-likelihood surface over (beta, gamma) ---
    # This is where the non-identifiability becomes visible: the high-likelihood
    # region is an elongated, tilted ridge running along roughly constant R0 = beta/gamma,
    # NOT a tight circular peak. This is *why* the inference is hard and *why* we need
    # robust samplers (SMC) / approximations (GP-HM).
    import matplotlib.pyplot as plt

    n_grid = 80
    beta_grid = np.linspace(0.35, 0.95, n_grid)
    gamma_grid = np.linspace(0.10, 0.35, n_grid)

    # Evaluate log-likelihood on the grid.
    # (Note: this is n_grid^2 = 6400 ODE solves -- a few seconds. It illustrates
    #  directly why a grid approach does NOT scale to higher dimensions: cost is
    #  n_grid^d, the curse of dimensionality.)
    LL = np.full((n_grid, n_grid), -np.inf)
    for ig, g in enumerate(gamma_grid):
        for ib, b in enumerate(beta_grid):
            LL[ig, ib] = log_likelihood(b, g, t_obs, data, SIGMA, y0=Y0)

    # For a readable contour plot, show the log-likelihood relative to its maximum,
    # and clip very low values (they'd otherwise dominate the colour scale).
    LL_rel = LL - np.nanmax(LL[np.isfinite(LL)])
    LL_rel = np.clip(LL_rel, -50, 0)  # only show within 50 log-units of the peak

    fig, ax = plt.subplots(figsize=(7, 6))
    BB, GG = np.meshgrid(beta_grid, gamma_grid)
    cf = ax.contourf(BB, GG, LL_rel, levels=30, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="log-likelihood (relative to maximum)")

    # Mark the TRUE parameters.
    ax.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=16,
            markeredgecolor="white", label="true (beta, gamma)")

    # Overlay a line of constant R0 = beta/gamma through the truth, to show the
    # ridge runs along it (the data constrains the RATIO well, the individual values poorly).
    R0_true = BETA_TRUE / GAMMA_TRUE
    gamma_line = np.linspace(gamma_grid[0], gamma_grid[-1], 100)
    beta_line = R0_true * gamma_line  # beta = R0 * gamma
    inside = (beta_line >= beta_grid[0]) & (beta_line <= beta_grid[-1])
    ax.plot(beta_line[inside], gamma_line[inside], "w--", lw=1.5,
            label=f"constant R0 = {R0_true:.1f}")

    ax.set_xlabel(r"$\beta$ (transmission rate)")
    ax.set_ylabel(r"$\gamma$ (recovery rate)")
    ax.set_title("Log-likelihood surface: the identifiability RIDGE")
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()
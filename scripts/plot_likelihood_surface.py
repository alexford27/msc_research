"""
Plot the SIR log-likelihood surface over (beta, gamma): the identifiability RIDGE.

This is the demo that previously lived in the __main__ block of the likelihood
module. It visualises *why* the inference is hard: the high-likelihood region is
an elongated, tilted ridge running along roughly constant R0 = beta/gamma, NOT a
tight circular peak. That non-identifiability is the motivation for robust
samplers (SMC) and approximations (GP history matching).

Run from the repo root:
    PYTHONPATH=src python scripts/plot_likelihood_surface.py
"""

import numpy as np
import matplotlib.pyplot as plt

from sir_inference.model.simulator import simulate_dataset
from sir_inference.model.likelihood import log_likelihood


if __name__ == "__main__":
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

    # --- Evaluate the log-likelihood on a (beta, gamma) grid ---
    # (Note: this is n_grid^2 = 6400 ODE solves -- a few seconds. It illustrates
    #  directly why a grid approach does NOT scale to higher dimensions: cost is
    #  n_grid^d, the curse of dimensionality.)
    n_grid = 80
    beta_grid = np.linspace(0.35, 0.95, n_grid)
    gamma_grid = np.linspace(0.10, 0.35, n_grid)

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

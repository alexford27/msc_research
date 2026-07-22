"""
DIAGNOSTIC: plot the emulator's predictive uncertainty (std) over the (beta, gamma)
grid, one panel per feature, plus the training points overlaid.

Purpose: check WHERE the emulator is confident vs uncertain. This helps explain
the shape of the non-implausible region -- e.g. if the region is wide at high
beta/gamma, is it because the emulator is UNCERTAIN there (large std -> inflated
implausibility denominator -> nothing ruled out), or because features genuinely
match the data there? This plot answers the "uncertain vs genuinely plausible"
question.

Recall: emulator predictive std is small near training points, large in
sparsely-sampled regions (it depends on WHERE the training points are).
"""

import numpy as np
import matplotlib.pyplot as plt

from sir_inference.inference.gp_hm.emulator import (
    FEATURE_NAMES, latin_hypercube_design, evaluate_design,
    fit_emulator, emulator_predict)


if __name__ == "__main__":
    T_EVAL = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)
    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2

    # --- Fit the emulator (same design as part 3) ---
    N_TRAIN = 100
    design = latin_hypercube_design(N_TRAIN, seed=0)
    train_features, _ = evaluate_design(design, T_EVAL, y0=Y0)
    emulators = fit_emulator(design, train_features)

    # --- Predict over a grid ---
    n_grid = 120
    beta_grid = np.linspace(0.0, 2.0, n_grid)
    gamma_grid = np.linspace(0.0, 1.0, n_grid)
    BB, GG = np.meshgrid(beta_grid, gamma_grid)
    grid_points = np.column_stack([BB.ravel(), GG.ravel()])

    means, stds = emulator_predict(emulators, grid_points)   # (Q,3),(Q,3)

    # --- Plot predictive std for each feature, with training points overlaid ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, (ax, name) in enumerate(zip(axes, FEATURE_NAMES)):
        std_grid = stds[:, j].reshape(n_grid, n_grid)
        cf = ax.contourf(BB, GG, std_grid, levels=30, cmap="magma")
        fig.colorbar(cf, ax=ax, label="predictive std")
        # overlay training points (where the emulator should be confident)
        ax.scatter(design[:, 0], design[:, 1], s=8, c="cyan",
                   edgecolors="black", linewidths=0.3, label="training points")
        ax.plot(BETA_TRUE, GAMMA_TRUE, "g*", markersize=15,
                markeredgecolor="white", label="truth")
        ax.set_title(f"Emulator uncertainty: {name}")
        ax.set_xlabel("beta"); ax.set_ylabel("gamma")
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Emulator predictive std over parameter space "
                 "(bright = uncertain, dark = confident)", fontsize=13)
    fig.tight_layout()
    plt.show()

    # --- Also print where uncertainty is highest/lowest, for quick reading ---
    print("Per-feature predictive std summary (over the grid):")
    print(f"{'feature':18s} {'min_std':>10s} {'max_std':>10s} {'std@truth':>10s}")
    _, std_truth = emulator_predict(emulators, np.array([[BETA_TRUE, GAMMA_TRUE]]))
    for j, name in enumerate(FEATURE_NAMES):
        print(f"{name:18s} {stds[:,j].min():10.4f} {stds[:,j].max():10.4f} "
              f"{std_truth[0, j]:10.4f}")
    print("\nIf std is large at high beta/gamma, the wide non-implausible region")
    print("there is driven by emulator UNCERTAINTY (not genuine fit) -- which waves")
    print("would fix by adding training points in the surviving region.")
"""
GP-HM sampler -- PART 4: WAVES (iterative history matching).

Wave 1 (parts 1+3) gives a broad, rough non-implausible region from a uniform
initial design. Waves refine it: each wave concentrates NEW training points
INSIDE the current non-implausible region, adds them to the training set, refits
a sharper emulator there, and recomputes implausibility -- shrinking the region.

The loop per wave:
  1. sample new candidate points INSIDE the current non-implausible region
  2. run the ODE there, extract features (the wave's cost)
  3. ADD them to the accumulated training set; REFIT the emulator
  4. recompute implausibility -> a tighter non-implausible region
Repeat for a few waves (or until the region stabilises).

IMPORTANT expectation (see earlier analysis): waves sharpen the EMULATOR, so they
tighten the region where emulator UNCERTAINTY was the limiter (across the ridge,
the boundary). They will NOT eliminate:
  - the elongation ALONG the R0 ridge (that's correct non-identifiability -- the
    features, being ~functions of R0, genuinely can't separate same-R0 params), nor
  - the flaring at high R0 (the summary features SATURATE there -- an information
    limit of the features, not the emulator).
So the correct target is a tighter RIDGE, not a tiny blob. The residual elongation
is HM faithfully reporting the same non-identifiability SMC shows.

Reuses parts 1 (emulator) and 3 (implausibility).
"""

import numpy as np

from sir_inference.inference.gp_hm.emulator import (
    FEATURE_NAMES, latin_hypercube_design, evaluate_design, fit_emulator)
from sir_inference.inference.gp_hm.implausability import (
    observed_features_and_noise, implausibility)
from sir_inference.inference.gp_hm.post_process import sample_uniform


def sample_in_region(emulators, z_obs, obs_noise_var, threshold,
                     n_wanted, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                     pool_factor=40, rng=None):
    """
    Draw n_wanted new training points spread INSIDE the current non-implausible
    region. Strategy: generate a large candidate pool over the box, keep those with
    I_max <= threshold (inside the region), then subsample n_wanted of them.

    Returns the chosen points (n_chosen, 2). n_chosen may be < n_wanted if the
    region is very small relative to the pool.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Large candidate pool over the whole box (cheap -- emulator eval, no ODE solves).
    n_pool = n_wanted * pool_factor
    pool = np.column_stack([
        rng.uniform(beta_range[0], beta_range[1], n_pool),
        rng.uniform(gamma_range[0], gamma_range[1], n_pool)])

    _, I_max = implausibility(emulators, pool, z_obs, obs_noise_var)
    inside = pool[I_max <= threshold]

    if len(inside) == 0:
        return np.empty((0, 2))
    if len(inside) <= n_wanted:
        return inside
    # Subsample for coverage (random subset; could use a space-filling subsample).
    idx = rng.choice(len(inside), size=n_wanted, replace=False)
    return inside[idx]


def run_waves(t_eval, beta_true, gamma_true, sigma, y0=(0.99, 0.01, 0.0),
              n_initial=100, n_per_wave=30, n_waves=3, threshold=3.0,
              grid_n=120, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
              seed=2327, verbose=True):
    """
    Run wave-based history matching. Returns a dict of results + per-wave history.
    """
    rng = np.random.default_rng(seed)

    # --- Observed features + observation-noise (fixed across waves) ---
    z_obs, obs_noise_var = observed_features_and_noise(
        beta_true, gamma_true, t_eval, sigma, y0=y0, data_seed=seed)

    # --- Wave 0: initial space-filling design ---
    design = latin_hypercube_design(n_initial, beta_range, gamma_range, seed=seed)
    features, n_solves = evaluate_design(design, t_eval, y0=y0)
    total_solves = n_solves
    emulators = fit_emulator(design, features)

    # Grid for evaluating / reporting the non-implausible fraction.
    beta_grid = np.linspace(beta_range[0], beta_range[1], grid_n)
    gamma_grid = np.linspace(gamma_range[0], gamma_range[1], grid_n)
    BB, GG = np.meshgrid(beta_grid, gamma_grid)
    grid_points = np.column_stack([BB.ravel(), GG.ravel()])

    def region_fraction_and_truth():
        _, I_max = implausibility(emulators, grid_points, z_obs, obs_noise_var)
        frac = np.mean(I_max <= threshold) * 100
        _, I_truth = implausibility(emulators, np.array([[beta_true, gamma_true]]),
                                    z_obs, obs_noise_var)
        return frac, I_truth[0], I_max.reshape(grid_n, grid_n)

    history = []
    frac, I_truth, I_grid = region_fraction_and_truth()
    history.append({"wave": 0, "n_train": len(design), "total_solves": total_solves,
                    "frac": frac, "I_truth": I_truth, "I_grid": I_grid})
    if verbose:
        print(f"{'wave':>4} {'n_train':>8} {'solves':>8} {'NI_frac%':>9} "
              f"{'I_truth':>8} {'truth_ok':>9}")
        print(f"{0:>4} {len(design):>8} {total_solves:>8} {frac:>9.1f} "
              f"{I_truth:>8.2f} {'yes' if I_truth <= threshold else 'NO!':>9}")

    # --- Waves 1..n_waves ---
    for w in range(1, n_waves + 1):
        # 1. sample new points inside the current non-implausible region
        new_pts = sample_in_region(emulators, z_obs, obs_noise_var, threshold,
                                   n_per_wave, beta_range, gamma_range, rng=rng)
        if len(new_pts) == 0:
            if verbose:
                print(f"  wave {w}: non-implausible region empty/tiny -- stopping.")
            break

        print(f"wave {w}: {len(new_pts)} new points, beta range [{new_pts[:, 0].min():.2f}, {new_pts[:, 0].max():.2f}]")

        # 2. evaluate the ODE at the new points (the wave's cost)
        new_feats, ns = evaluate_design(new_pts, t_eval, y0=y0)
        total_solves += ns

        # 3. accumulate and refit
        design = np.vstack([design, new_pts])
        features = np.vstack([features, new_feats])
        emulators = fit_emulator(design, features)

        # 4. recompute region
        frac, I_truth, I_grid = region_fraction_and_truth()
        history.append({"wave": w, "n_train": len(design),
                        "total_solves": total_solves, "frac": frac,
                        "I_truth": I_truth, "I_grid": I_grid})
        if verbose:
            print(f"{w:>4} {len(design):>8} {total_solves:>8} {frac:>9.1f} "
                  f"{I_truth:>8.2f} {'yes' if I_truth <= threshold else 'NO!':>9}")

    return {"emulators": emulators, "design": design, "features": features,
            "z_obs": z_obs, "obs_noise_var": obs_noise_var,
            "history": history, "total_solves": total_solves,
            "grid": (BB, GG), "threshold": threshold}


# ----------------------------------------------------------------------
# DEMO: run waves and plot the region shrinking
# ------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_EVAL = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    result = run_waves(T_EVAL, BETA_TRUE, GAMMA_TRUE, SIGMA, y0=Y0,
                       n_initial=100, n_per_wave=30, n_waves=3, threshold=3.0, seed=2327)

    hm_samples = sample_uniform(result["emulators"], result["z_obs"],
                                result["obs_noise_var"], threshold=3.0,
                                n_samples=10000, rng=np.random.default_rng(0))



    np.savez("hm_run.npz",
             samples=hm_samples,
             total_solves=result["total_solves"],
             z_obs=result["z_obs"],
             obs_noise_var=result["obs_noise_var"])

    # --- CONSISTENCY CHECK: same dataset as the SMC run? ---
    from sir_inference.model.simulator import simulate_dataset

    _, _, data_check, _ = simulate_dataset(BETA_TRUE, GAMMA_TRUE, T_EVAL, SIGMA,
                                           y0=Y0, observed=("s", "i", "r"), seed=2327)
    smc = np.load("smc_run.npz")
    for c in ("s", "i", "r"):
        assert np.allclose(data_check[c], smc[f"data_{c}"]), \
            f"DATASET MISMATCH on '{c}' -- HM and SMC are fitting different data!"
    print("Dataset consistency check: PASSED (HM and SMC use identical data)")

    BB, GG = result["grid"]
    history = result["history"]
    design = result["design"]
    thr = result["threshold"]

    print(f"\nFinal: {result['total_solves']} total ODE solves "
          f"(vs ~25,000 for SMC -- the cost contrast).")

    print(f"\nFinal: {result['total_solves']} total ODE solves "
          f"(vs 17,655 for SMC -- the cost contrast).")

    # --- DIAGNOSTIC: what dominates the implausibility denominator? ---
    from sir_inference.inference.gp_hm.emulator import emulator_predict, FEATURE_NAMES
    from sir_inference.inference.gp_hm.implausability import implausibility

    emulators = result["emulators"]
    z_obs = result["z_obs"]
    obs_noise_var = result["obs_noise_var"]
    BB_, GG_ = result["grid"]
    grid_points = np.column_stack([BB_.ravel(), GG_.ravel()])

    _, I_grid_pts = implausibility(emulators, grid_points, z_obs, obs_noise_var)
    inside = grid_points[I_grid_pts <= result["threshold"]]
    print(f"\nVariance decomposition inside the non-implausible region "
          f"({len(inside)} grid points):")

    _, stds_in = emulator_predict(emulators, inside)
    emu_var = stds_in ** 2

    print(f"{'feature':18s} {'mean emu_var':>14s} {'obs_noise_var':>14s} {'ratio':>8s}")
    for j, name in enumerate(FEATURE_NAMES):
        e = emu_var[:, j].mean();
        o = obs_noise_var[j]
        print(f"{name:18s} {e:14.3e} {o:14.3e} {e / o:8.3f}")
    print("\n  ratio << 1 -> observation-noise-limited (waves cannot help)")
    print("  ratio ~ 1+ -> emulator matters; refinement should have worked")

    # --- Plot: non-implausible region per wave + accumulated training points ---
    n_panels = len(history)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    axes = axes[0]
    g_line = np.linspace(GG.min() + 1e-3, GG.max(), 100)
    for ax, h in zip(axes, history):
        non_imp = h["I_grid"] <= thr
        ax.contourf(BB, GG, non_imp, levels=[0.5, 1.5], colors=["tab:green"], alpha=0.5)
        ax.plot((BETA_TRUE / GAMMA_TRUE) * g_line, g_line, "k--", lw=1, label="R0=3")
        ax.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=14,
                markeredgecolor="white", label="truth")
        # training points present by this wave: first n_train of accumulated design
        n_tr = h["n_train"]
        ax.scatter(design[:n_tr, 0], design[:n_tr, 1], s=6, c="navy", alpha=0.4)
        ax.set_xlim(BB.min(), BB.max()); ax.set_ylim(GG.min(), GG.max())
        ax.set_title(f"Wave {h['wave']}: NI={h['frac']:.1f}%, n={n_tr}")
        ax.set_xlabel("beta"); ax.set_ylabel("gamma")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("History matching waves: non-implausible region shrinking "
                 "(green), training points (navy)", fontsize=13)
    fig.tight_layout()
    plt.show()
"""
GP-HM sampler -- PART 3: IMPLAUSIBILITY and the non-implausible region.

Given the fitted emulator (part 1), this computes the implausibility measure
over parameter space and identifies the non-implausible region -- the set of
(beta, gamma) NOT ruled out by the data.

Implausibility for feature k at parameter theta:
    I_k(theta) = | z_obs_k - E[f_k(theta)] | / sqrt( Var_k )
where
    Var_k = emulator_variance_k(theta)      (the GP predictive variance at theta)
          + obs_noise_variance_k            (how trajectory noise propagates to feature k)
          + model_discrepancy_k             (= 0 here: simulated data, model IS truth)

Combine across features by taking the MAX (a parameter must be consistent with
ALL features to survive):  I_max(theta) = max_k I_k(theta).
Rule out theta if I_max(theta) > 3 (the 3-sigma rule). The survivors are the
NON-IMPLAUSIBLE REGION.

Observation-noise variance (Option A, empirical): the observed features are
themselves uncertain because the observed trajectory is noisy. We estimate this
by re-drawing the observation noise on the truth's trajectory many times,
extracting features each time, and taking the variance of each feature. (Cheap:
no ODE solves -- we re-noise ONE clean trajectory repeatedly.)

At 2D we can grid implausibility densely (the validation regime). Wave-based
active refinement (for when gridding won't scale) is part 4.

Depends on part 1 (emulator) and the simulator.
"""

import numpy as np

from sir_inference.model.simulator import solve_sir, simulate_dataset
from sir_inference.inference.gp_hm.emulator import (
    extract_features, FEATURE_NAMES, latin_hypercube_design,
    evaluate_design, fit_emulator, emulator_predict)


# ----------------------------------------------------------------------
# Observed features + observation-noise variance (Option A)
# ----------------------------------------------------------------------

def observed_features_and_noise(beta_true, gamma_true, t_eval, sigma,
                                y0=(0.99, 0.01, 0.0), observed=("s", "i", "r"),
                                data_seed=0, n_noise=2000, rng=None):
    """
    Compute the observed feature values z_obs (from the actual noisy dataset), and
    estimate the observation-noise variance per feature (Option A, empirical).

    z_obs: features extracted from the ONE observed noisy dataset (the data we
           actually condition on).

    obs_noise_var: how much the features wobble due to observation noise. Estimated
           by re-drawing the sigma noise on the CLEAN truth trajectory n_noise times,
           extracting features each time, and taking the variance of each feature.
           This captures how trajectory-level noise (same sigma everywhere) propagates
           into each feature DIFFERENTLY (peak vs peak-time vs final-size).

    Returns
    -------
    z_obs        : array (3,)   observed feature values
    obs_noise_var: array (3,)   observation-noise variance per feature
    """
    if rng is None:
        rng = np.random.default_rng(data_seed)

    # The actual observed dataset (what we condition on) -- one noisy realisation.
    t_obs, truth_traj, data, meta = simulate_dataset(
        beta_true, gamma_true, t_eval, sigma, y0=y0, observed=observed, seed=data_seed)
    # `data` is a dict {compartment: noisy array}. extract_features expects a (3, T)
    # array [s; i; r], so assemble one. For any compartment NOT observed, fall back
    # to the clean truth row (so features remain computable under partial observation;
    # for the full-observation validation case all three are present anyway).
    index = {"s": 0, "i": 1, "r": 2}
    data_array = np.array(truth_traj, dtype=float).copy()   # start from truth (3, T)
    for name, noisy in data.items():
        data_array[index[name]] = noisy                     # overwrite observed rows
    z_obs = extract_features(t_obs, data_array)

    # --- Estimate observation-noise variance per feature (Option A) ---
    # Take the CLEAN truth trajectory and re-noise it many times.
    clean = solve_sir(beta_true, gamma_true, t_eval, y0=y0)   # (3, T), noise-free
    feats = np.empty((n_noise, 5))
    for m in range(n_noise):
        noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
        feats[m] = extract_features(t_eval, noisy)
    obs_noise_var = np.var(feats, axis=0)     # variance of each feature across re-noisings

    return z_obs, obs_noise_var


# ----------------------------------------------------------------------
# Implausibility over a set of query points
# ----------------------------------------------------------------------

def implausibility(emulators, query_points, z_obs, obs_noise_var,
                   model_discrepancy_var=None):
    """
    Compute per-feature and max implausibility at each query (beta, gamma).

    Parameters
    ----------
    emulators        : fitted GPs (one per feature) from part 1
    query_points     : array (Q, 2)
    z_obs            : array (3,)  observed feature values
    obs_noise_var    : array (3,)  observation-noise variance per feature
    model_discrepancy_var : array (3,) or None  (None -> zeros; simulated data)

    Returns
    -------
    I_per_feature : array (Q, 3)   implausibility for each feature
    I_max         : array (Q,)     max implausibility across features
    """
    means, stds = emulator_predict(emulators, query_points)   # (Q,F),(Q,F)
    emulator_var = stds ** 2

    if model_discrepancy_var is None:
        model_discrepancy_var = np.zeros(means.shape[1])   # one per feature

    # Total variance in the denominator: emulator + observation + model-discrepancy.
    total_var = emulator_var + obs_noise_var[None, :] + model_discrepancy_var[None, :]

    # Standardised mismatch per feature.
    mismatch = np.abs(z_obs[None, :] - means)
    I_per_feature = mismatch / np.sqrt(total_var)

    # Combine across features: a parameter must be consistent with ALL -> take MAX.
    I_max = np.max(I_per_feature, axis=1)
    return I_per_feature, I_max


# ----------------------------------------------------------------------
# DEMO: implausibility over a grid + non-implausible region plot
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_EVAL = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)
    THRESHOLD = 3.0

    # --- 1. Build and fit the emulator (part 1) ---
    N_TRAIN = 100
    design = latin_hypercube_design(N_TRAIN, seed=0)
    train_features, _ = evaluate_design(design, T_EVAL, y0=Y0)
    emulators = fit_emulator(design, train_features)
    print(f"Emulator fitted on {N_TRAIN} points.")

    # --- 2. Observed features + observation-noise variance (Option A) ---
    z_obs, obs_noise_var = observed_features_and_noise(
        BETA_TRUE, GAMMA_TRUE, T_EVAL, SIGMA, y0=Y0, data_seed=0)
    print("\nObserved features (z_obs):")
    for name, z, v in zip(FEATURE_NAMES, z_obs, obs_noise_var):
        print(f"  {name:18s}: z_obs={z:8.4f}   obs_noise_std={np.sqrt(v):.4f}")
    # Note how the observation-noise std DIFFERS across features -- the same trajectory
    # noise (sigma=0.1) propagates into each feature by a different amount.

    # --- 3. Implausibility over a dense grid (2D validation regime) ---
    n_grid = 120
    beta_grid = np.linspace(0.0, 2.0, n_grid)
    gamma_grid = np.linspace(0.0, 1.0, n_grid)
    BB, GG = np.meshgrid(beta_grid, gamma_grid)
    grid_points = np.column_stack([BB.ravel(), GG.ravel()])

    I_per_feature, I_max = implausibility(emulators, grid_points, z_obs, obs_noise_var)
    I_max_grid = I_max.reshape(n_grid, n_grid)

    # Non-implausible region: I_max <= threshold
    non_implausible = I_max_grid <= THRESHOLD
    frac = np.mean(non_implausible) * 100
    print(f"\nNon-implausible region: {frac:.1f}% of the parameter box "
          f"(I_max <= {THRESHOLD}).")
    # Is the truth non-implausible? (it should be)
    _, I_truth = implausibility(emulators, np.array([[BETA_TRUE, GAMMA_TRUE]]),
                                z_obs, obs_noise_var)
    print(f"Implausibility at the TRUTH: I_max = {I_truth[0]:.2f}  "
          f"({'NON-implausible (good)' if I_truth[0] <= THRESHOLD else 'RULED OUT (bad!)'})")

    # --- 4. Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Implausibility surface (capped for colour clarity)
    cf = ax1.contourf(BB, GG, np.clip(I_max_grid, 0, 10), levels=30, cmap="viridis_r")
    fig.colorbar(cf, ax=ax1, label="max implausibility (capped at 10)")
    ax1.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=16, markeredgecolor="white",
             label="truth")
    ax1.set_title("Implausibility surface")
    ax1.set_xlabel("beta"); ax1.set_ylabel("gamma"); ax1.legend()

    # (b) Non-implausible region (binary: in or out)
    ax2.contourf(BB, GG, non_implausible, levels=[0.5, 1.5], colors=["tab:green"],
                 alpha=0.5)
    ax2.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=16, markeredgecolor="white",
             label="truth")
    g_line = np.linspace(gamma_grid[1], gamma_grid[-1], 100)
    ax2.plot((BETA_TRUE / GAMMA_TRUE) * g_line, g_line, "k--", lw=1.2,
             label="R0=3 ridge")
    ax2.set_xlim(0, 2); ax2.set_ylim(0, 1)
    ax2.set_title(f"Non-implausible region (I_max <= {THRESHOLD})")
    ax2.set_xlabel("beta"); ax2.set_ylabel("gamma"); ax2.legend()

    fig.tight_layout()
    plt.show()
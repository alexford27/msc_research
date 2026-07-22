"""
GP-HM sampler -- PART 1: the GP EMULATOR.

The approximate arm. Instead of evaluating the expensive ODE everywhere (as SMC
does), we run it at a MODEST, cleverly-chosen set of (beta, gamma) points, extract
a few SUMMARY FEATURES of each trajectory, and fit a Gaussian Process to each
feature. The fitted GPs are a cheap surrogate ("emulator") that predicts each
feature -- WITH UNCERTAINTY -- at any new (beta, gamma), no ODE solve required.

What we emulate (low-dimensional summary features, the standard/tractable route):
  SHAPE features (functions of R0):    peak infectious fraction, final epidemic size
  TIMESCALE features (break R0-degeneracy): early growth rate, i at two fixed times
The shape features pin down R0; the timescale features pin down position along the
constant-R0 ridge -- together they constrain beta and gamma separately.

Each feature gets its OWN scalar GP:  (beta, gamma) -> feature value (+ uncertainty).
(A multi-output GP could share information across features -- the optional advanced
route -- but independent scalar GPs are the standard, tractable starting point.)

Design: a space-filling (Latin Hypercube) set of training points, so the emulator
sees the whole parameter box with few runs. (Active wave-based refinement comes later.)

Library: scikit-learn GaussianProcessRegressor (handles marginal-likelihood
hyperparameter fitting and target normalisation for us). We build the
HISTORY-MATCHING logic ourselves (later parts); the GP fitting is generic
infrastructure, so we borrow it.

Depends on solve_sir (the simulator).
"""

import numpy as np
from scipy.stats import qmc                      # quasi-Monte-Carlo: Latin Hypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from sir_inference.model.simulator import solve_sir   # adjust import to your layout


# ----------------------------------------------------------------------
# SUMMARY FEATURES of an SIR trajectory
# ----------------------------------------------------------------------

def extract_features(t, traj):
    """
    Reduce a full SIR trajectory to summary features.

    Parameters
    ----------
    t    : array (T,)      time points
    traj : array (3, T)    [s; i; r] over time (rows s, i, r)

    Returns
    -------
    features : array (5,)  [peak_i, final_size, growth_rate, i_at_t1, i_at_t2]

    FEATURE DESIGN -- two complementary kinds:

    SHAPE features (functions of R0 = beta/gamma; constant ALONG the ridge):
      - peak_i      : maximum infectious fraction
      - final_size  : final recovered fraction (total ever infected)
      These constrain R0 well but CANNOT separate (beta, gamma) pairs with the
      same R0 -- so on their own they give a region elongated along the ridge.

    TIMESCALE features (depend on beta, gamma SEPARATELY; vary ALONG the ridge):
      - growth_rate : early exponential growth rate, estimated as the slope of
                      log(i) over the early (rising) phase. For small initial
                      infected fraction, r ~ beta - gamma, which depends on the
                      DIFFERENCE of the rates, not their ratio -- so it varies
                      along constant-R0 and BREAKS the R0-degeneracy.
      - i_at_t1, i_at_t2 : the infectious fraction at two fixed early times.
                      A faster epidemic (larger absolute rates, same R0) is further
                      along at any fixed time, so these also capture timescale and
                      help separate beta from gamma. Smooth functions of (beta,gamma),
                      unlike the erratic argmax-based time-of-peak (which was dropped).

    Together: SHAPE features pin down R0, TIMESCALE features pin down the position
    ALONG the ridge -> the two kinds jointly constrain both beta and gamma.

    Degenerate cases (R0 < 1, no epidemic): peak_i ~ initial i, final_size tiny,
    growth_rate negative (i decays), i_at_t small -- all still well-defined.
    """
    s, i, r = traj[0], traj[1], traj[2]
    t = np.asarray(t, dtype=float)

    # --- SHAPE features ---
    peak_i = np.max(i)
    final_size = r[-1]

    # --- TIMESCALE feature 1: early exponential growth rate ---
    # Fit a straight line to log(i) over the early phase (before the peak), whose
    # slope is the exponential growth rate. Use points up to the peak (or a capped
    # early window), guarding against log(0).
    peak_idx = int(np.argmax(i))
    # early window: from start up to (but not beyond) the peak, at least 2 points
    end = max(2, min(peak_idx + 1, len(t)))
    i_early = np.clip(i[:end], 1e-12, None)        # avoid log(0)
    t_early = t[:end]
    if end >= 2 and np.ptp(t_early) > 0:
        # slope of log(i) vs t (least-squares line) = growth rate
        growth_rate = np.polyfit(t_early, np.log(i_early), 1)[0]
    else:
        growth_rate = 0.0

    # --- TIMESCALE features 2 & 3: infectious fraction at two fixed early times ---
    # Interpolate i at chosen times (smooth, unlike argmax timing). Times picked in
    # the rising/early phase where same-R0 epidemics most differ in speed.
    T1, T2 = 8.0, 16.0
    i_at_t1 = float(np.interp(T1, t, i))
    i_at_t2 = float(np.interp(T2, t, i))

    return np.array([peak_i, final_size, growth_rate, i_at_t1, i_at_t2])


# Names, for plots/printing.
FEATURE_NAMES = ["peak_infectious", "final_size", "growth_rate",
                 "i_at_t8", "i_at_t16"]


# ----------------------------------------------------------------------
# INPUT SCALING
# ----------------------------------------------------------------------
# The GP inputs (beta, gamma) live on DIFFERENT ranges (beta in [0,2], gamma in
# [0,1]). Unequal input ranges make the ARD length-scale optimisation poorly
# conditioned -- the marginal-likelihood surface is awkward and lbfgs fails to
# converge (the "scale the data" warning). Fixing this means scaling both inputs
# to a common [0,1] range before fitting/predicting. The parameter box is fixed
# throughout the project, so we scale with these known constants (no scaler to
# pass around). normalize_y handles the TARGETS; this handles the INPUTS.
BETA_BOX = (0.0, 2.0)
GAMMA_BOX = (0.0, 1.0)


def _scale_inputs(points):
    """Map (beta, gamma) from the parameter box to [0,1]^2 (for the GP)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] - BETA_BOX[0]) / (BETA_BOX[1] - BETA_BOX[0])
    out[:, 1] = (points[:, 1] - GAMMA_BOX[0]) / (GAMMA_BOX[1] - GAMMA_BOX[0])
    return out


# ----------------------------------------------------------------------
# SPACE-FILLING DESIGN: where to run the ODE
# ----------------------------------------------------------------------

def latin_hypercube_design(n_points, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                           seed=None):
    """
    Latin Hypercube design over the (beta, gamma) box: spreads n_points evenly
    across parameter space (better coverage than a uniform random scatter for the
    same number of points). Returns array (n_points, 2).
    """
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n=n_points)            # points in [0,1]^2
    lower = [beta_range[0], gamma_range[0]]
    upper = [beta_range[1], gamma_range[1]]
    return qmc.scale(unit, lower, upper)         # scale to the actual box


# ----------------------------------------------------------------------
# Run the ODE at design points and extract features (the EXPENSIVE step)
# ----------------------------------------------------------------------

def evaluate_design(design, t_eval, y0=(0.99, 0.01, 0.0), rtol=1e-5, atol=1e-7):
    """
    For each (beta, gamma) in the design, solve the ODE and extract features.
    This is the emulator's training cost: n_points ODE solves.

    Returns
    -------
    features : array (n_points, n_features)   summary features per design point
    n_solves : int
    """
    n = design.shape[0]
    n_features = len(FEATURE_NAMES)
    features = np.empty((n, n_features))
    for k in range(n):
        beta, gamma = design[k]
        traj = solve_sir(beta, gamma, t_eval, y0=y0, rtol=rtol, atol=atol)
        features[k] = extract_features(t_eval, traj)
    return features, n


# ----------------------------------------------------------------------
# FIT the emulator: one GP per feature
# ----------------------------------------------------------------------

def fit_emulator(design, features, n_restarts=5):
    """
    Fit one GP per summary feature, mapping (beta, gamma) -> feature.

    Kernel: Constant * Matern(nu=2.5) + White.
      - Matern(nu=2.5): smooth but less rigidly so than RBF -- a sensible default
        for emulating a physical model output (RBF's infinite smoothness is often
        too strong). ARD (separate length-scale per input) via length_scale as a
        2-vector, so beta and gamma can have different relevant scales.
      - ConstantKernel: learns the output variance (amplitude).
      - WhiteKernel: a small noise term for numerical stability / jitter (the ODE
        output is deterministic, but a tiny nugget stabilises the fit).
    Inputs are SCALED to [0,1]^2 before fitting (see _scale_inputs) so the ARD
    length-scale optimisation is well-conditioned -- this is what prevents the
    lbfgs convergence failures. Length-scale start (0.3) and bounds suit [0,1] inputs.
    normalize_y=True centres/scales each feature (so the zero-mean GP is appropriate).

    Returns
    -------
    emulators : list of fitted GaussianProcessRegressor (one per feature)
    """
    design_scaled = _scale_inputs(design)
    emulators = []
    for j in range(features.shape[1]):
        kernel = (ConstantKernel(1.0, (1e-3, 1e5))
                  * Matern(length_scale=[0.3, 0.3],
                           length_scale_bounds=(1e-2, 1e1), nu=2.5)
                  + WhiteKernel(noise_level=1e-6,
                                noise_level_bounds=(1e-8, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                      n_restarts_optimizer=n_restarts)
        gp.fit(design_scaled, features[:, j])
        emulators.append(gp)
    return emulators


def emulator_predict(emulators, query_points):
    """
    Predict all features at query (beta, gamma) points, with uncertainty.

    Returns
    -------
    means : array (n_query, n_features)   predictive mean of each feature
    stds  : array (n_query, n_features)   predictive std (uncertainty) of each feature
    """
    query_points = np.atleast_2d(query_points)
    query_scaled = _scale_inputs(query_points)        # same scaling as fitting
    n_q = query_points.shape[0]
    n_features = len(emulators)
    means = np.empty((n_q, n_features))
    stds = np.empty((n_q, n_features))
    for j, gp in enumerate(emulators):
        m, sd = gp.predict(query_scaled, return_std=True)
        means[:, j] = m
        stds[:, j] = sd
    return means, stds


# ----------------------------------------------------------------------
# DEMO: build emulator, validate on held-out points
# ----------------------------------------------------------------------

if __name__ == "__main__":
    T_EVAL = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    # --- Build a training design and evaluate it (the expensive step) ---
    N_TRAIN = 40
    design = latin_hypercube_design(N_TRAIN, seed=0)
    train_features, n_solves = evaluate_design(design, T_EVAL, y0=Y0)
    print(f"Training design: {N_TRAIN} points, {n_solves} ODE solves.")

    # --- Fit the emulator (one GP per feature) ---
    emulators = fit_emulator(design, train_features)
    print("\nFitted kernels (per feature):")
    for name, gp in zip(FEATURE_NAMES, emulators):
        print(f"  {name:18s}: {gp.kernel_}")

    # --- VALIDATE on a fresh held-out set: does the emulator predict well? ---
    N_TEST = 30
    test_design = latin_hypercube_design(N_TEST, seed=999)   # different seed
    test_features, _ = evaluate_design(test_design, T_EVAL, y0=Y0)
    pred_means, pred_stds = emulator_predict(emulators, test_design)

    print("\nHeld-out validation (per feature):")
    print(f"{'feature':18s} {'RMSE':>10s} {'mean|err|':>10s} "
          f"{'mean_std':>10s} {'in_2std%':>9s}")
    for j, name in enumerate(FEATURE_NAMES):
        true = test_features[:, j]
        pred = pred_means[:, j]
        err = pred - true
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(np.abs(err))
        mean_std = np.mean(pred_stds[:, j])
        # calibration: fraction of true values within +/- 2 predictive std of pred
        within = np.mean(np.abs(err) <= 2 * pred_stds[:, j]) * 100
        print(f"{name:18s} {rmse:10.4f} {mae:10.4f} {mean_std:10.4f} {within:8.0f}%")

    print("\nInterpretation:")
    print(" - low RMSE/MAE  -> emulator predicts the features accurately")
    print(" - in_2std% near 95 -> predictive uncertainty is well-calibrated")
    print("   (too low: overconfident; too high: underconfident)")
    print(" If accuracy is poor, add more training points (increase N_TRAIN).")
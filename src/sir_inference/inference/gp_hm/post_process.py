"""
GP-HM sampler -- PART 5: POST-PROCESSING the non-implausible region into a
posterior-comparable DISTRIBUTION (Objective 3).

WHY: SMC produces a DISTRIBUTION (a weighted particle cloud -- graded, denser
where more probable). History matching produces a SET (the non-implausible region
-- binary in/out, no gradation). To COMPARE the two methods we need them in the
same currency, so we turn the HM set into a distribution we can sample from. Then
both methods are "clouds of points" and directly comparable (same plots, same
distance metrics, same coverage checks).

The hard part (the open question -- Marina Q5b): HM DISCARDED the gradation (it
only kept in/out), so reconstructing a distribution RE-INTRODUCES information HM
threw away. There is no uniquely "correct" way. We implement two:

  OPTION 1 -- UNIFORM over the non-implausible set:
      every point with I_max <= threshold is equally probable, zero outside.
      The honest LITERAL representation of HM's output (a set, no internal
      gradation). Crude as a posterior: hard edges, no peak. The natural baseline.

  OPTION 2 -- IMPLAUSIBILITY-WEIGHTED:
      grade points by fit quality, weight ~ exp(-0.5 * D^2), where D^2 combines the
      per-feature implausibilities. This RECOVERS gradation (peaked near the best
      fit, tailing off) -- much closer in SHAPE to an SMC posterior. But the map
      from implausibility to density is a MODELLING CHOICE (not unique), so it needs
      justification. We use the SUM of squared per-feature implausibilities
      (D^2 = sum_k I_k^2), which is the Gaussian-(log-)likelihood analogue over the
      features -- smoother and more likelihood-like than exponentiating I_max.
      (The ruling-out / Option-1 region still uses I_max, consistent with HM.)

  OPTION 3 -- emulator-implied likelihood (most principled, blurs toward ABC):
      deferred pending Marina's steer.

Reuses parts 1 (emulator) and 3 (implausibility), and consumes the result dict
returned by part 4's run_waves (the REFINED final-wave emulator).
"""

import numpy as np

from sir_inference.inference.gp_hm.emulator import emulator_predict
from sir_inference.inference.gp_hm.implausability import implausibility


# ----------------------------------------------------------------------
# OPTION 1: uniform samples over the non-implausible set
# ----------------------------------------------------------------------

def sample_uniform(emulators, z_obs, obs_noise_var, threshold=3.0, n_samples=2000,
                   beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                   pool_factor=25, rng=None, max_tries=20):
    """
    Draw n_samples uniformly from {theta : I_max(theta) <= threshold} by rejection:
    sample candidates over the box, keep those inside the region, repeat until we
    have enough. Returns (n_samples, 2). Each sample is equally weighted.
    """
    if rng is None:
        rng = np.random.default_rng()
    kept = []
    tries = 0
    while sum(len(k) for k in kept) < n_samples and tries < max_tries:
        n_pool = n_samples * pool_factor
        pool = np.column_stack([
            rng.uniform(*beta_range, n_pool),
            rng.uniform(*gamma_range, n_pool)])
        _, I_max = implausibility(emulators, pool, z_obs, obs_noise_var)
        kept.append(pool[I_max <= threshold])
        tries += 1
    allpts = np.vstack(kept) if kept else np.empty((0, 2))
    if len(allpts) < n_samples:
        return allpts                                  # region too small; return what we have
    idx = rng.choice(len(allpts), size=n_samples, replace=False)
    return allpts[idx]


# ----------------------------------------------------------------------
# OPTION 2: implausibility-weighted samples
# ----------------------------------------------------------------------

def sample_weighted(emulators, z_obs, obs_noise_var, n_samples=2000,
                    threshold=3.0, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                    pool_factor=200, rng=None):
    """
    Draw n_samples from a distribution weighted by fit quality, RESTRICTED to the
    non-implausible region:
        weight(theta) ~ exp(-0.5 * D^2(theta))   for theta with I_max <= threshold,
        weight = 0                                otherwise.
        D^2 = sum_k I_k^2.

    WHY restrict to the region: implausibility puts the emulator variance in its
    DENOMINATOR, so where the emulator is UNCERTAIN (large variance) the
    implausibility is small -> exp(-0.5 D^2) is non-negligible -> spurious high
    weight far from the truth (uncertainty masquerading as probability). Masking to
    I_max <= threshold first removes that artifact: we grade by fit WITHIN what HM
    has not ruled out, and assign zero weight to ruled-out parameters. This is the
    principled combination of Option 1 (the set) and the gradation of Option 2.

    Method: importance-style. Uniform pool -> keep I_max <= threshold -> weight by
    exp(-0.5 D^2) -> resample n_samples (with replacement). Yields a PEAKED cloud
    comparable in shape to an SMC posterior. Returns (n_samples, 2).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_pool = n_samples * pool_factor
    pool = np.column_stack([
        rng.uniform(*beta_range, n_pool),
        rng.uniform(*gamma_range, n_pool)])

    # Restrict to the non-implausible region FIRST (removes uncertainty-driven
    # spurious far points).
    I_per_feature, I_max = implausibility(emulators, pool, z_obs, obs_noise_var)
    inside = I_max <= threshold
    pool = pool[inside]
    if len(pool) == 0:
        return np.empty((0, 2))
    D2 = np.sum(I_per_feature[inside] ** 2, axis=1)

    # weights ~ exp(-0.5 D^2); subtract max log-weight for numerical stability
    # (constant cancels under normalisation -- same max-trick idea as in SMC).
    logw = -0.5 * D2
    logw -= logw.max()
    w = np.exp(logw)
    w_sum = w.sum()
    if w_sum <= 0 or not np.isfinite(w_sum):
        return np.empty((0, 2))
    w /= w_sum
    idx = rng.choice(len(pool), size=n_samples, replace=True, p=w)
    return pool[idx]


# ----------------------------------------------------------------------
# Summary stats (to eyeball against SMC)
# ----------------------------------------------------------------------

def summarise(samples, label=""):
    """Print posterior-style summary (mean, std of beta, gamma, R0) for a cloud."""
    if len(samples) == 0:
        print(f"{label}: EMPTY (no samples).")
        return
    beta, gamma = samples[:, 0], samples[:, 1]
    R0 = beta / np.clip(gamma, 1e-9, None)
    print(f"{label}  (n={len(samples)})")
    print(f"    beta : mean={beta.mean():.3f}  std={beta.std():.3f}")
    print(f"    gamma: mean={gamma.mean():.3f}  std={gamma.std():.3f}")
    print(f"    R0   : mean={R0.mean():.3f}  std={R0.std():.3f}")


# ----------------------------------------------------------------------
# DEMO: post-process a waves result, compare the two options
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sir_inference.inference.gp_hm.waves import run_waves

    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_EVAL = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)
    THRESHOLD = 3.0

    # --- Run waves to get the refined emulator (part 4) ---
    result = run_waves(T_EVAL, BETA_TRUE, GAMMA_TRUE, SIGMA, y0=Y0,
                       n_initial=100, n_per_wave=30, n_waves=2, threshold=THRESHOLD,
                       verbose=False)
    emulators = result["emulators"]
    z_obs = result["z_obs"]
    obs_noise_var = result["obs_noise_var"]
    print(f"Post-processing the refined emulator "
          f"({result['total_solves']} ODE solves).\n")

    rng = np.random.default_rng(0)

    # --- Option 1: uniform over the non-implausible set ---
    samp_uniform = sample_uniform(emulators, z_obs, obs_noise_var,
                                  threshold=THRESHOLD, n_samples=2000, rng=rng)
    summarise(samp_uniform, "OPTION 1  uniform-over-set")

    # --- Option 2: implausibility-weighted (restricted to non-implausible region) ---
    samp_weighted = sample_weighted(emulators, z_obs, obs_noise_var,
                                    n_samples=2000, threshold=THRESHOLD, rng=rng)
    summarise(samp_weighted, "OPTION 2  implausibility-weighted")

    print(f"\n(For comparison, SMC posterior was ~ beta=0.58, R0=3.36.)")

    # --- Plot the two clouds side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    g_line = np.linspace(0.01, 1.0, 100)
    for ax, samp, title in [
            (axes[0], samp_uniform, "Option 1: uniform over non-implausible set"),
            (axes[1], samp_weighted, "Option 2: implausibility-weighted")]:
        if len(samp):
            ax.scatter(samp[:, 0], samp[:, 1], s=4, alpha=0.2, c="tab:blue")
        ax.plot((BETA_TRUE / GAMMA_TRUE) * g_line, g_line, "k--", lw=1, label="R0=3")
        ax.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=15,
                markeredgecolor="white", label="truth")
        ax.set_xlim(0, 2); ax.set_ylim(0, 1)
        ax.set_title(title); ax.set_xlabel("beta"); ax.set_ylabel("gamma")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("GP-HM post-processed posteriors (compare shape to SMC)", fontsize=13)
    fig.tight_layout()
    plt.show()
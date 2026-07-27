"""
Generate and FREEZE the 5D SEIR dataset.

Run once. Every 5D component (SMC, emulator/HM, scores, grid) must load this
file rather than regenerating from a seed: seed-based regeneration is fragile,
since any change to call order or library version silently changes the data,
and a dataset mismatch between arms invalidates the whole comparison.

DESIGN CHOICES (recorded here so they are not lost)
---------------------------------------------------
Observation model: ALL FOUR compartments (s, e, i, r).

  Chosen so that the dimensional comparison is CONTROLLED. Moving from 2D to 5D
  should change one thing -- the number of parameters. If observability changed
  too (e.g. to infectious-only), any degradation observed could not be attributed
  to dimension rather than to seeing less of the system. The 2D tier observed all
  its compartments, so the 5D tier does the same.

  The cost of this choice: observing the exposed compartment is unrealistic --
  exposed-but-not-yet-infectious individuals are not observable in practice. Full
  observation therefore makes 5D EASIER than a realistic problem would be, so any
  degradation measured here is a LOWER BOUND on what the approximate arm would
  suffer in application. Infectious-only observation is the natural robustness
  check, and the machinery supports it by changing one argument.

Truth: (beta, sigma_E, gamma, e0, i0) = (0.6, 0.2, 0.2, 0.005, 0.005)
  R0 = beta/gamma = 3, matching the 2D study for continuity; mean latent period
  1/sigma_E = 5 days; 0.5% initially exposed and 0.5% initially infectious.

Timing: 40 observations over 80 days. Longer than the 2D window (60 days)
  because the latent compartment delays the epidemic -- the peak occurs around
  day 35, so 60 days would truncate the decay phase and lose the information
  that identifies gamma.

Noise: sigma = 0.05, half the 2D value. SEIR's compartments are individually
  smaller (the infected are split across e and i), so the same absolute noise
  would swamp more of the signal.
"""

import numpy as np

from seir_score import (simulate_dataset, solve_seir, COMPARTMENT_INDEX,
                        PARAM_NAMES, default_prior)


# --- Configuration (single source of truth for the 5D tier) ---
THETA_TRUE = np.array([0.6, 0.2, 0.2, 0.005, 0.005])
SIGMA = 0.05
T_OBS = np.linspace(0, 80, 40)
OBSERVED = ("s", "e", "i", "r")          # all four -- see note above
SEED = 2327                              # same seed convention as the 2D tier
OUT = "seir_data.npz"


def main():
    truth, data = simulate_dataset(THETA_TRUE, T_OBS, SIGMA,
                                   observed=OBSERVED, seed=SEED)

    print("=" * 70)
    print("5D SEIR DATASET")
    print("=" * 70)
    for name, val in zip(PARAM_NAMES, THETA_TRUE):
        print(f"  {name:9s} = {val}")
    print(f"  R0        = {THETA_TRUE[0] / THETA_TRUE[2]:.2f}")
    print(f"  latent    = {1 / THETA_TRUE[1]:.1f} days")
    print(f"  sigma     = {SIGMA}")
    print(f"  observed  = {OBSERVED}")
    print(f"  times     = {len(T_OBS)} points over [{T_OBS[0]:.0f}, {T_OBS[-1]:.0f}] days")

    # --- Sanity checks on the trajectory ---
    i_traj = truth[COMPARTMENT_INDEX["i"]]
    e_traj = truth[COMPARTMENT_INDEX["e"]]
    sums = truth.sum(axis=0)
    print(f"\n  conservation : sum in [{sums.min():.8f}, {sums.max():.8f}]")
    print(f"  peak i       : {i_traj.max():.4f} at t = {T_OBS[i_traj.argmax()]:.0f} d")
    print(f"  peak e       : {e_traj.max():.4f} at t = {T_OBS[e_traj.argmax()]:.0f} d")
    print(f"  final r      : {truth[COMPARTMENT_INDEX['r']][-1]:.4f}")

    # Is the observation window long enough to capture the decay?
    tail = i_traj[-1] / max(i_traj.max(), 1e-12)
    print(f"  i at end / peak i = {tail:.3f}  "
          f"({'window captures the decay' if tail < 0.1 else 'WARNING: window may truncate the epidemic'})")

    # Is the noise comparable to the 2D tier? (Not an absolute threshold -- the
    # point is that the dimensional comparison stays controlled, so 5D should sit
    # at roughly the same signal-to-noise as the completed 2D study.)
    # 2D reference: sigma=0.10, peak i = 0.299  ->  noise/peak = 0.33
    ratio = SIGMA / i_traj.max()
    print(f"  noise/peak-i = {ratio:.2f}   (2D tier ran at 0.33)  "
          f"{'-> comparable, dimensional comparison controlled' if abs(ratio - 0.33) < 0.10 else '-> DIFFERS from 2D; degradation would be confounded with noise'}")

    np.savez(OUT,
             theta_true=THETA_TRUE, t_obs=T_OBS, sigma=SIGMA,
             truth=truth, observed=np.array(OBSERVED), seed=SEED,
             **{f"data_{c}": v for c, v in data.items()})
    print(f"\nSaved -> {OUT}")
    print("Every 5D component should load this file, not regenerate from the seed.")


if __name__ == "__main__":
    main()
"""
SMC sampler -- PART 2: systematic resampling.

Resampling resets a degenerated weighted population back to EQUAL weights:
draw N new particles in proportion to the current weights (high-weight particles
duplicated, low-weight ones die), then reset all weights to uniform. This is what
stops the weight degeneration you saw in part 1 -- each tempering step ends with a
healthy equal-weight population, so the next adaptive step starts from full ESS.

SYSTEMATIC resampling (the standard choice): lowest variance, fastest, uses ONE
random number. Lay the normalised weights end-to-end as a cumulative distribution
on [0,1]; draw one offset u in [0, 1/N]; place N evenly-spaced pointers at
u, u+1/N, u+2/N, ...; each pointer selects whichever particle's cumulative interval
it falls in.

Slots into the loop AFTER reweighting: reweight -> check ESS -> (if low) resample
-> [move, part 3].
"""

import numpy as np


def systematic_resample(particles, log_weights, rng=None):
    """
    Systematic resampling.

    Parameters
    ----------
    particles : array (N, d)        current particle locations
    log_weights : array (N,)        current UNnormalised log-weights
    rng : np.random.Generator

    Returns
    -------
    new_particles : array (N, d)    resampled particles (with duplicates), N same as input
    new_log_weights : array (N,)    reset to equal weights (all zeros = log 1, unnormalised)
    indices : array (N,)            which original particle each new one came from
                                    (useful for resampling auxiliary arrays, e.g. loglikes)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = particles.shape[0]

    # --- Normalise weights (stably) to a probability distribution ---
    m = np.max(log_weights)
    w = np.exp(log_weights - m)
    W = w / np.sum(w)                       # normalised weights, sum to 1

    # --- Cumulative distribution of the weights ---
    cumulative = np.cumsum(W)
    cumulative[-1] = 1.0                     # guard against tiny floating-point drift

    # --- N evenly-spaced pointers, offset by ONE random draw in [0, 1/N) ---
    offset = rng.uniform(0.0, 1.0 / n)
    pointers = offset + np.arange(n) / n     # u, u+1/N, u+2/N, ...

    # --- For each pointer, find which particle's cumulative interval it lands in ---
    # searchsorted finds, for each pointer, the index of the first cumulative value
    # >= the pointer -> that's the selected particle.
    indices = np.searchsorted(cumulative, pointers)
    # (clip just in case of edge floating-point: indices must be valid 0..n-1)
    indices = np.clip(indices, 0, n - 1)

    new_particles = particles[indices].copy()
    new_log_weights = np.zeros(n)            # reset to EQUAL weights (log 1)

    return new_particles, new_log_weights, indices


# ----------------------------------------------------------------------
# STANDALONE TEST: verify resampling behaves correctly (no SMC needed)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Make a small, deliberately UNEVEN weighted population to test on ---
    # 5 particles; particle 0 has most of the weight, 3 and 4 almost none.
    particles = np.array([[0.60, 0.20],     # particle 0  (should be duplicated a lot)
                          [0.58, 0.19],      # particle 1
                          [0.62, 0.21],      # particle 2
                          [1.50, 0.05],      # particle 3  (should mostly die)
                          [0.30, 0.80]])     # particle 4  (should mostly die)
    # Log-weights corresponding to normalised weights ~ [0.60, 0.25, 0.10, 0.03, 0.02]
    W_target = np.array([0.60, 0.25, 0.10, 0.03, 0.02])
    log_weights = np.log(W_target)

    def ess(logw):
        m = np.max(logw); w = np.exp(logw - m); W = w / w.sum()
        return 1.0 / np.sum(W ** 2)

    print("BEFORE resampling:")
    print(f"  normalised weights: {W_target}")
    print(f"  ESS = {ess(log_weights):.2f}  (out of 5)")

    # --- Resample ---
    new_particles, new_log_weights, indices = systematic_resample(
        particles, log_weights, rng=rng)

    print("\nAFTER resampling:")
    print(f"  selected indices: {indices}")
    print(f"  (particle 0 should appear ~3x, particles 3&4 should mostly vanish)")
    print(f"  new weights all equal: {np.allclose(new_log_weights, new_log_weights[0])}")
    print(f"  ESS = {ess(new_log_weights):.2f}  (should be back to 5 = full)")

    # --- Check it preserves the distribution on average over many resamples ---
    # Count how often each particle is selected across many independent resamples;
    # the average count should match N * its weight.
    counts = np.zeros(5)
    n_trials = 20000
    for _ in range(n_trials):
        _, _, idx = systematic_resample(particles, log_weights,
                                        rng=np.random.default_rng())
        for j in idx:
            counts[j] += 1
    avg_copies = counts / n_trials
    print("\nAverage copies per particle over many resamples:")
    print(f"  observed: {np.round(avg_copies, 3)}")
    print(f"  expected (N * weight): {np.round(5 * W_target, 3)}")
    print("  --> these should match: resampling preserves the distribution on average.")
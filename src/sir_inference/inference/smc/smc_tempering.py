"""
SMC sampler for the SIR posterior -- PART 1:tempe ring schedule + reweighting.

This builds the CORE of the SMC sampler incrementally. Part 1 covers:
  - initialise particles from the prior (uniform box)
  - the ADAPTIVE tempering schedule: choose each next phi so that post-reweight
    ESS hits a target (e.g. N/2), via a free 1D bisection search that reuses
    already-computed per-particle log-likelihoods (NO new ODE solves)
  - the reweight step (incremental weight = likelihood^(delta phi))

NOT yet included (later parts): resampling, the RWMH move kernel, the full loop
that produces the posterior, and the evidence Z byproduct.

Likelihood tempering:  pi_phi(theta) ∝ prior(theta) * likelihood(theta)^phi,
with phi going 0 (prior) -> 1 (posterior).

Key efficiency: each particle's log-likelihood is computed ONCE per phi-level
(in the reweight). The adaptive phi-search only re-powers those stored values,
so it costs no extra ODE solves.

Depends on log_likelihood and log_prior (which depend on solve_sir).
"""

import numpy as np

from sir_inference.inference.smc.smc_resampling import systematic_resample
from sir_inference.model.likelihood import log_likelihood          # adjust import to your layout



# ----------------------------------------------------------------------
# ESS helper
# ----------------------------------------------------------------------

def effective_sample_size(log_weights):
    """
    ESS = 1 / sum(W_i^2) for normalised weights W.
    Computed stably from UNnormalised log-weights.
    """
    # Normalise in log space (subtract max for stability), exponentiate.
    m = np.max(log_weights)
    w = np.exp(log_weights - m)
    w_sum = np.sum(w)
    W = w / w_sum                      # normalised weights
    return 1.0 / np.sum(W ** 2)


# ----------------------------------------------------------------------
# Initialise particles from the prior (uniform box)
# ----------------------------------------------------------------------

def init_particles(n_particles, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                   rng=None):
    """
    Draw N particles uniformly from the prior box. Returns array (N, 2).
    """
    if rng is None:
        rng = np.random.default_rng()
    betas = rng.uniform(beta_range[0], beta_range[1], size=n_particles)
    gammas = rng.uniform(gamma_range[0], gamma_range[1], size=n_particles)
    return np.column_stack([betas, gammas])


# ----------------------------------------------------------------------
# Compute per-particle log-likelihoods (the expensive step: N ODE solves)
# ----------------------------------------------------------------------

def compute_loglikes(particles, t_obs, data, sigma, y0=(0.99, 0.01, 0.0)):
    """
    Log-likelihood for each particle. This is the ONE expensive call per phi-level
    (N ODE solves). The adaptive phi-search below reuses these, free.

    Returns array (N,) of log-likelihoods (-inf for invalid/failed particles).
    """
    n = particles.shape[0]
    loglikes = np.empty(n)
    for i in range(n):
        loglikes[i] = log_likelihood(particles[i, 0], particles[i, 1],
                                     t_obs, data, sigma, y0=y0)
    return loglikes


# ----------------------------------------------------------------------
# The adaptive tempering step: find next phi so post-reweight ESS = target
# ----------------------------------------------------------------------

def next_phi(phi_current, loglikes, current_log_weights, ess_target,
             tol=1e-6, min_step=1e-3):
    """
    Find the next temperature phi_new in (phi_current, 1] such that the ESS of
    the reweighted particles equals ess_target. If even phi_new = 1 keeps ESS
    above the target, return 1.0 (final step).

    The incremental log-weight for stepping phi_current -> phi_new is
        delta_logw_i = (phi_new - phi_current) * loglike_i
    added to the current log-weights. ESS decreases monotonically as phi_new
    grows, so we bisect.

    A MINIMUM step (min_step) is enforced: the search bracket starts at
    phi_current + min_step rather than phi_current. This guarantees the schedule
    always advances (no degenerate zero-steps, which can otherwise occur right
    after a resample when the equal-weight population's ESS sits exactly at the
    target). min_step is tiny relative to the schedule, so it doesn't distort the
    adaptive behaviour -- it just stops the loop stalling.

    NO ODE solves here -- purely arithmetic on stored loglikes.
    """

    def ess_at(phi_new):
        delta = phi_new - phi_current
        new_log_weights = current_log_weights + delta * loglikes
        return effective_sample_size(new_log_weights)

    # If jumping straight to phi=1 keeps ESS above target, we're done.
    if ess_at(1.0) >= ess_target:
        return 1.0

    # Enforce a minimum forward step: smallest phi considered is
    # phi_current + min_step (capped at 1.0). Guarantees progress.
    phi_floor = min(phi_current + min_step, 1.0)

    # If even the minimum step already drops ESS to/below target, take it
    # (the target wants a step smaller than min_step; we take min_step to keep moving).
    if ess_at(phi_floor) <= ess_target:
        return phi_floor

    # Otherwise bisection search for phi_new where ESS(phi_new) = ess_target,
    # in the bracket (phi_floor, 1].
    lo, hi = phi_floor, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) > ess_target:
            lo = mid  # ESS too high -> need bigger step -> increase phi
        else:
            hi = mid  # ESS too low -> step too big -> decrease phi
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# DEMO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from sir_inference.model.simulator import simulate_dataset

    # --- Data (sigma=0.1 case) ---
    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)
    t_obs, truth, data, meta = simulate_dataset(
        BETA_TRUE, GAMMA_TRUE, T_OBS, SIGMA, y0=Y0,
        observed=("s", "i", "r"), seed=0)

    # --- SMC settings ---
    N = 1000
    ESS_TARGET = N / 2
    ESS_THRESHOLD = N / 2  # resample when ESS drops to/below this
    rng = np.random.default_rng(0)

    # --- Initialise: particles from prior, equal (log) weights ---
    particles = init_particles(N, rng=rng)
    log_weights = np.zeros(N)                 # equal weights -> log 0 (unnormalised)
    phi = 0.0

    print(f"Initialised {N} particles from prior. ESS target = {ESS_TARGET:.0f}")
    print(f"{'step':>4} {'phi':>8} {'delta_phi':>10} {'ESS':>8}")

    # NOTE: in part 1 we do NOT resample or move, so we recompute loglikes once
    # from the initial particles and just walk the schedule to show it behaving.
    # (In the full loop, particles change each step via resample+move and loglikes
    #  are recomputed -- that's parts 2 and 3.)
    loglikes = compute_loglikes(particles, t_obs, data, SIGMA, y0=Y0)

    step = 0
    while phi < 1.0:
        step += 1
        phi_new = next_phi(phi, loglikes, log_weights, ESS_TARGET)
        print(f"   DEBUG: phi_current={phi:.6f}, phi_new={phi_new:.6f}, "
              f"ess_at_1={effective_sample_size(log_weights + (1.0 - phi) * loglikes):.1f}, "
              f"unique_particles={len(np.unique(particles, axis=0))}")
        delta = phi_new - phi

        # Reweight: add incremental log-weight.
        log_weights = log_weights + delta * loglikes
        ess = effective_sample_size(log_weights)

        # resample when ESS drops to/below target
        ess_threshold = 0.95 * N  # rather than 0.5 * N
        if ess <= ess_threshold:
            particles, log_weights, indices = systematic_resample(particles, log_weights, rng=rng)
            loglikes = loglikes[indices]

        print(f"{step:>4} {phi_new:>8.4f} {delta:>10.4f} {ess:>8.1f}")

        phi = phi_new


    print(f"\nReached phi = 1 in {step} tempering steps.")
    print("phi-schedule + reweight mechanics. Adding resample+move keeps ESS healthy")
    print("and produces the actual posterior.")
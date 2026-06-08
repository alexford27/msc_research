import numpy as np

from sir_inference.inference.smc.smc_move import move_particles
from sir_inference.inference.smc.smc_resampling import systematic_resample
from sir_inference.inference.smc.smc_tempering import (
    effective_sample_size, init_particles, compute_loglikes, next_phi)


def smc_sampler(t_obs, data, sigma, n_particles=1000, ess_frac=0.5,
                n_move_steps=5, cov_scale=1.0, y0=(0.99, 0.01, 0.0),
                beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
                rng=None, verbose=True):
    """
    Full adaptive-tempering SMC sampler for the SIR posterior.

    Loop per tempering step:
      1. choose next phi (free ESS-targeting search on stored loglikes)
      2. reweight to the new phi
      3. resample if ESS <= threshold (reset to equal weights, realign loglikes)
      4. MOVE each particle (RWMH targeting pi_phi) -> restores diversity,
         recomputes loglikes
    until phi reaches 1.

    Returns a dict with the final particles (the posterior sample), plus diagnostics
    including the total ODE-solve count (the hardware-independent cost metric) and
    the log-evidence estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = n_particles
    ess_threshold = ess_frac * N

    # --- Initialise from prior, equal weights, phi = 0 ---
    particles = init_particles(N, beta_range, gamma_range, rng=rng)
    log_weights = np.zeros(N)
    phi = 0.0

    # initial loglikes (N solves)
    loglikes = compute_loglikes(particles, t_obs, data, sigma, y0=y0)
    total_solves = N

    log_evidence = 0.0          # accumulates log of the normaliser ratios (the Z estimate)
    step = 0

    if verbose:
        print(f"SMC: {N} particles, ESS threshold {ess_threshold:.0f}, "
              f"{n_move_steps} move steps/particle")
        print(f"{'step':>4} {'phi':>8} {'d_phi':>8} {'ESS':>7} "
              f"{'accept':>7} {'unique':>7} {'solves':>9}")

    while phi < 1.0:
        step += 1

        # 1. choose next phi (free)
        phi_new = next_phi(phi, loglikes, log_weights, ess_threshold)
        delta = phi_new - phi

        # --- accumulate log-evidence: log of mean incremental weight ---
        # incremental log-weights for this step = delta * loglikes (added to current)
        # The evidence contribution is log( sum_i W_i * exp(delta*loglike_i) ) using
        # the CURRENT normalised weights W_i. (Stable log-sum-exp.)
        inc = delta * loglikes
        m = np.max(log_weights)
        W = np.exp(log_weights - m); W /= W.sum()        # current normalised weights
        log_evidence += np.log(np.sum(W * np.exp(inc - np.max(inc)))) + np.max(inc)

        # 2. reweight
        log_weights = log_weights + inc
        ess = effective_sample_size(log_weights)
        phi = phi_new

        # 3. resample if degenerated
        if ess <= ess_threshold:
            particles, log_weights, indices = systematic_resample(particles, log_weights, rng=rng)
            loglikes = loglikes[indices]

        # 4. MOVE -- adapt proposal covariance from the current particle cloud
        cloud_cov = np.cov(particles.T)
        proposal_cov = cov_scale * cloud_cov + 1e-10 * np.eye(2)   # jitter for safety
        particles, loglikes, accept, n_solves = move_particles(
            particles, loglikes, phi, t_obs, data, sigma, proposal_cov,
            n_move_steps, y0=y0, beta_range=beta_range, gamma_range=gamma_range, rng=rng)
        total_solves += n_solves

        if verbose:
            n_unique = len(np.unique(particles, axis=0))
            print(f"{step:>4} {phi:>8.4f} {delta:>8.4f} {ess:>7.1f} "
                  f"{accept:>7.3f} {n_unique:>7d} {total_solves:>9d}")

    if verbose:
        print(f"\nDone: phi=1 in {step} steps, {total_solves:,} total ODE solves.")
        print(f"Posterior mean: beta={particles[:,0].mean():.3f}, "
              f"gamma={particles[:,1].mean():.3f}  (truth 0.6, 0.2)")
        R0 = particles[:, 0] / particles[:, 1]
        print(f"R0 posterior:   mean={R0.mean():.2f}, std={R0.std():.2f}  (truth 3.0)")
        print(f"log-evidence estimate: {log_evidence:.2f}")

    return {
        "particles": particles,
        "log_weights": log_weights,
        "n_steps": step,
        "total_solves": total_solves,
        "log_evidence": log_evidence,
    }


# ----------------------------------------------------------------------
# The COMPLETE SMC sampler
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# DEMO: run the full sampler and validate against the RWMH posterior
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sir_inference.model.simulator import simulate_dataset

    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)
    t_obs, truth, data, meta = simulate_dataset(
        BETA_TRUE, GAMMA_TRUE, T_OBS, SIGMA, y0=Y0,
        observed=("s", "i", "r"), seed=2327)

    rng = np.random.default_rng(0)
    result = smc_sampler(t_obs, data, SIGMA, n_particles=1000,
                         ess_frac=0.5, n_move_steps=3, cov_scale=1.0, rng=rng)

    particles = result["particles"]

    # --- Plot the SMC posterior over the ridge ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(particles[:, 0], particles[:, 1], s=5, alpha=0.2, color="tab:blue",
               label="SMC particles")
    ax.plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=16, markeredgecolor="white",
            label="truth")
    g_line = np.linspace(0.1, 0.35, 50)
    ax.plot((BETA_TRUE / GAMMA_TRUE) * g_line, g_line, "k--", lw=1, label="R0=3")
    ax.set_xlabel("beta"); ax.set_ylabel("gamma")
    ax.set_title(f"SMC posterior (sigma={SIGMA}, {result['n_steps']} steps, "
                 f"{result['total_solves']:,} solves)")
    ax.legend()
    fig.tight_layout()
    plt.show()

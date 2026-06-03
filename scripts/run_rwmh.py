"""
Random-Walk Metropolis-Hastings (RWMH) for the SIR Bayesian inverse problem.

This is BOTH:
  - the baseline sampler to compare SMC against (Objective 2), and
  - the move kernel that lives INSIDE SMC.

It samples the POSTERIOR  p(beta, gamma | data)  proportional to  prior x likelihood.

Design (all deliberate; see methodology notes):
  - Works entirely in LOG space (log-posterior = log-prior + log-likelihood);
    the unknown evidence Z cancels in the MH ratio, so we only need the
    unnormalised log-posterior.
  - SYMMETRIC Gaussian random-walk proposal -> proposal ratio cancels ->
    acceptance simplifies to  min{1, post(theta')/post(theta)},
    i.e. in log space: accept if  log(U) < logpost(theta') - logpost(theta).
  - UNIFORM-BOX prior on positive ranges: log-prior = 0 inside the box, -inf
    outside. This also enforces positivity automatically (proposals outside the
    box are rejected), matching the parameter guard in the likelihood.
  - Tracks the acceptance rate so the step size can be tuned (target ~0.234).

Depends on log_likelihood (which itself depends on solve_sir).
"""

import numpy as np
from sir_inference.model.likelihood import log_likelihood


# ----------------------------------------------------------------------
# THE PRIOR  (uniform box on positive ranges)
# ----------------------------------------------------------------------

def log_prior(beta, gamma, beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0)):
    """
    Log of a uniform prior on a positive box.

    Returns 0.0 inside the box (uniform density -> constant -> 0 in log, up to a
    constant that cancels in the MH ratio anyway), and -inf outside (zero prior
    probability). The -inf outside also enforces positivity, so invalid parameters
    are rejected automatically.
    """
    if beta_range[0] < beta < beta_range[1] and gamma_range[0] < gamma < gamma_range[1]:
        return 0.0
    return -np.inf


# ----------------------------------------------------------------------
# THE LOG-POSTERIOR  (prior x likelihood, in log space)
# ----------------------------------------------------------------------

def log_posterior(beta, gamma, t_obs, data, sigma,
                  y0=(0.99, 0.01, 0.0),
                  beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0)):
    """
    Unnormalised log-posterior = log-prior + log-likelihood.

    Short-circuits: if the prior is -inf (outside the box), we don't even bother
    solving the ODE -- the point is rejected regardless. This saves expensive
    ODE solves on out-of-bounds proposals.
    """
    lp = log_prior(beta, gamma, beta_range, gamma_range)
    if not np.isfinite(lp):
        return -np.inf                      # outside prior support -> skip the solve
    return lp + log_likelihood(beta, gamma, t_obs, data, sigma, y0=y0)


# ----------------------------------------------------------------------
# THE RWMH SAMPLER
# ----------------------------------------------------------------------

def rwmh(t_obs, data, sigma,
         theta0, n_iter,
         proposal_cov,
         y0=(0.99, 0.01, 0.0),
         beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0),
         rng=None,
         progress_every=2000):
    """
    Random-Walk Metropolis-Hastings sampling of the SIR posterior.

    Parameters
    ----------
    t_obs, data, sigma : the observed dataset and known noise level
    theta0 : array (2,)        starting point [beta, gamma]
    n_iter : int               number of iterations
    proposal_cov : array (2,2) covariance of the Gaussian random-walk proposal
                               (the tuning knob: controls step size/shape)
    y0 : initial conditions (fixed)
    beta_range, gamma_range : uniform-prior box
    rng : np.random.Generator  for reproducibility
    progress_every : int       print a progress line every this many iterations
                               (set to 0 or None to silence)

    Returns
    -------
    chain : array (n_iter+1, 2)   the sampled (beta, gamma) values (incl. start)
    logposts : array (n_iter+1,)  log-posterior at each chain state
    accept_rate : float           fraction of proposals accepted
    """
    import time

    if rng is None:
        rng = np.random.default_rng()

    theta0 = np.asarray(theta0, dtype=float)
    proposal_cov = np.asarray(proposal_cov, dtype=float)

    # Pre-allocate storage for the chain and its log-posterior values.
    chain = np.zeros((n_iter + 1, 2))
    logposts = np.zeros(n_iter + 1)

    # Initialise.
    chain[0] = theta0
    logposts[0] = log_posterior(theta0[0], theta0[1], t_obs, data, sigma,
                                y0=y0, beta_range=beta_range, gamma_range=gamma_range)
    if not np.isfinite(logposts[0]):
        raise ValueError("Starting point has zero posterior (outside prior or "
                         "un-solvable). Choose a valid theta0.")

    n_accepted = 0
    current = theta0.copy()
    current_logpost = logposts[0]

    start_time = time.time()

    # --- The main MH loop ---
    for k in range(1, n_iter + 1):
        # 1. PROPOSE: symmetric Gaussian step centred on the current point.
        proposal = rng.multivariate_normal(current, proposal_cov)

        # 2. EVALUATE the log-posterior at the proposal.
        proposal_logpost = log_posterior(
            proposal[0], proposal[1], t_obs, data, sigma,
            y0=y0, beta_range=beta_range, gamma_range=gamma_range)

        # 3. ACCEPT / REJECT.
        #    Symmetric proposal -> accept with prob min{1, exp(logpost' - logpost)}.
        #    In log space: accept if log(U) < logpost' - logpost.
        log_accept_ratio = proposal_logpost - current_logpost
        if np.log(rng.uniform()) < log_accept_ratio:
            current = proposal
            current_logpost = proposal_logpost
            n_accepted += 1
        # else: reject -> stay at current (chain repeats the current point).

        chain[k] = current
        logposts[k] = current_logpost

        # --- Progress report ---
        if progress_every and (k % progress_every == 0):
            elapsed = time.time() - start_time
            rate = k / elapsed                      # iterations per second
            remaining = (n_iter - k) / rate         # estimated seconds left
            running_accept = n_accepted / k
            print(f"  iter {k:>7,}/{n_iter:,}  "
                  f"({100*k/n_iter:4.0f}%)  "
                  f"accept={running_accept:.3f}  "
                  f"current=({current[0]:.3f}, {current[1]:.3f})  "
                  f"{rate:.0f} it/s  "
                  f"~{remaining:.0f}s left")

    total_time = time.time() - start_time
    accept_rate = n_accepted / n_iter
    if progress_every:
        print(f"  done: {n_iter:,} iterations in {total_time:.1f}s "
              f"({n_iter/total_time:.0f} it/s), final accept rate {accept_rate:.3f}")
    return chain, logposts, accept_rate


# ----------------------------------------------------------------------
# DEMO  (run this file directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sir_inference.model.simulator import simulate_dataset

    # --- Generate data (use the harder sigma=0.1 case to see the ridge) ---
    BETA_TRUE, GAMMA_TRUE = 0.6, 0.2
    SIGMA = 0.1
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    t_obs, truth, data, meta = simulate_dataset(
        BETA_TRUE, GAMMA_TRUE, T_OBS, SIGMA,
        y0=Y0, observed=("s", "i", "r"), seed=0)

    # --- Run RWMH ---
    rng = np.random.default_rng(42)
    theta0 = np.array([0.5, 0.25])          # deliberately start away from the truth
    n_iter = 20000
    # Proposal covariance: the tuning knob. Start isotropic & smallish; tune below.
    step = 0.02
    proposal_cov = (step ** 2) * np.eye(2)

    print(f"Running RWMH: {n_iter:,} iterations (~{n_iter:,} ODE solves)...")
    chain, logposts, accept_rate = rwmh(
        t_obs, data, SIGMA, theta0, n_iter, proposal_cov, y0=Y0, rng=rng,
        progress_every=2000)

    print(f"Acceptance rate: {accept_rate:.3f}  (aim ~0.2-0.4; tune `step`)")

    # --- Discard burn-in (the initial transient before the chain settles) ---
    burn_in = 2000
    samples = chain[burn_in:]

    # --- Posterior summaries ---
    print(f"\nPosterior mean:  beta={samples[:,0].mean():.3f}, "
          f"gamma={samples[:,1].mean():.3f}   (truth: 0.6, 0.2)")
    print(f"Posterior std :  beta={samples[:,0].std():.3f}, "
          f"gamma={samples[:,1].std():.3f}")
    # R0 posterior (computed per-sample, then summarised) -- the well-identified quantity
    R0_samples = samples[:, 0] / samples[:, 1]
    print(f"R0 posterior  :  mean={R0_samples.mean():.2f}, "
          f"std={R0_samples.std():.2f}   (truth: 3.0)")

    # --- PLOTS ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Trace plots: each parameter over iterations (check mixing & burn-in)
    axes[0, 0].plot(chain[:, 0], lw=0.5)
    axes[0, 0].axhline(BETA_TRUE, color="r", ls="--", label="truth")
    axes[0, 0].axvline(burn_in, color="grey", ls=":", label="burn-in cutoff")
    axes[0, 0].set_title("Trace: beta");  axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("beta");        axes[0, 0].legend()

    axes[0, 1].plot(chain[:, 1], lw=0.5, color="tab:orange")
    axes[0, 1].axhline(GAMMA_TRUE, color="r", ls="--", label="truth")
    axes[0, 1].axvline(burn_in, color="grey", ls=":", label="burn-in cutoff")
    axes[0, 1].set_title("Trace: gamma"); axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("gamma");       axes[0, 1].legend()

    # (b) The chain exploring the (beta, gamma) plane -- should trace the RIDGE
    axes[1, 0].plot(samples[:, 0], samples[:, 1], lw=0.3, alpha=0.5, color="tab:blue")
    axes[1, 0].plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=15,
                    markeredgecolor="white", label="truth")
    # constant-R0 line for reference
    g_line = np.linspace(0.1, 0.35, 50)
    axes[1, 0].plot((BETA_TRUE / GAMMA_TRUE) * g_line, g_line, "k--", lw=1,
                    label="constant R0=3")
    axes[1, 0].set_title("Chain path in (beta, gamma) -- traces the ridge")
    axes[1, 0].set_xlabel("beta"); axes[1, 0].set_ylabel("gamma")
    axes[1, 0].legend()

    # (c) Posterior samples as a scatter/density (the actual posterior estimate)
    axes[1, 1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.1, color="tab:blue")
    axes[1, 1].plot(BETA_TRUE, GAMMA_TRUE, "r*", markersize=15,
                    markeredgecolor="white", label="truth")
    axes[1, 1].set_title("Posterior samples (after burn-in)")
    axes[1, 1].set_xlabel("beta"); axes[1, 1].set_ylabel("gamma")
    axes[1, 1].legend()

    fig.suptitle(f"RWMH on SIR posterior  (sigma={SIGMA}, "
                 f"accept rate={accept_rate:.2f})", fontsize=14)
    fig.tight_layout()
    plt.show()
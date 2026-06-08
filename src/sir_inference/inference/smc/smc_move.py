"""
SMC sampler -- PART 3: the MCMC move step, and the COMPLETE assembled sampler.

The move step is what cures both problems you saw in part 2:
  - sample IMPOVERISHMENT (resampling makes duplicates; nothing spreads them)
  - the schedule STUTTER (a degenerate duplicated population breaks next_phi)

After resampling, each particle takes a few RWMH steps targeting the CURRENT
tempered distribution  pi_phi ∝ prior * likelihood^phi. Duplicates start together
but take different random moves, so they spread apart -- diversity restored --
while the MCMC steps preserve the target (they're valid moves for pi_phi).

Key design choices:
  - Move targets the TEMPERED posterior (prior + phi*loglike), not the full one.
  - Proposal covariance is ADAPTED from the current particle cloud's covariance
    (scaled): broad moves at low phi, tight moves at high phi -- automatically.
    This is exactly the adaptivity that fixes RWMH's manual-tuning fragility.
  - After moving, each particle's loglike is RECOMPUTED at its new location.
    This is the dominant cost: ~ N * n_move_steps ODE solves per tempering step.

Reuses: solve_sir, log_likelihood, log_prior, and the part-1 helpers.
"""

import numpy as np

from sir_inference.model.likelihood import log_likelihood        # adjust to your layout
from sir_inference.inference.rwmh import log_prior                    # uniform-box prior


# ----------------------------------------------------------------------
# Tempered log-posterior (the move's target at temperature phi)
# ----------------------------------------------------------------------

def tempered_log_posterior(beta, gamma, phi, loglike_value,
                           beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0)):
    """
    log pi_phi(theta) = log_prior(theta) + phi * loglike(theta)   (up to a constant).

    Takes the already-computed loglike_value to avoid a redundant ODE solve when
    we already know it (e.g. for the current point). The prior enforces the box
    (and positivity) by returning -inf outside.
    """
    lp = log_prior(beta, gamma, beta_range, gamma_range)
    if not np.isfinite(lp):
        return -np.inf
    return lp + phi * loglike_value


# ----------------------------------------------------------------------
# The MOVE step: RWMH on each particle, targeting pi_phi
# ----------------------------------------------------------------------

def move_particles(particles, loglikes, phi, t_obs, data, sigma,
                   proposal_cov, n_steps, y0=(0.99, 0.01, 0.0),
                   beta_range=(0.0, 2.0), gamma_range=(0.0, 1.0), rng=None):
    """
    Move each particle with n_steps of RWMH targeting the tempered posterior pi_phi.

    Returns
    -------
    new_particles : array (N, 2)   moved particle locations
    new_loglikes  : array (N,)     loglike recomputed at the moved locations
    accept_rate   : float          overall move acceptance rate (a diagnostic)
    n_solves      : int            ODE solves used (the move's cost)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = particles.shape[0]
    new_particles = particles.copy()
    new_loglikes = loglikes.copy()
    n_accept = 0
    n_solves = 0

    for i in range(n):
        theta = new_particles[i].copy()
        ll = new_loglikes[i]                       # current loglike (already known)
        logpost = tempered_log_posterior(theta[0], theta[1], phi, ll,
                                         beta_range, gamma_range)

        for _ in range(n_steps):
            # propose
            prop = rng.multivariate_normal(theta, proposal_cov)

            # cheap prior check first -- skip the ODE solve if out of the box
            lp = log_prior(prop[0], prop[1], beta_range, gamma_range)
            if not np.isfinite(lp):
                continue                            # reject, no solve needed

            # compute loglike at proposal (ONE ODE solve)
            ll_prop = log_likelihood(prop[0], prop[1], t_obs, data, sigma, y0=y0)
            n_solves += 1
            logpost_prop = lp + phi * ll_prop

            # accept/reject (log space)
            if np.log(rng.uniform()) < logpost_prop - logpost:
                theta = prop
                ll = ll_prop
                logpost = logpost_prop
                n_accept += 1

        new_particles[i] = theta
        new_loglikes[i] = ll

    accept_rate = n_accept / (n * n_steps)
    return new_particles, new_loglikes, accept_rate, n_solves

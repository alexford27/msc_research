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
         rng=None):
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

    Returns
    -------
    chain : array (n_iter+1, 2)   the sampled (beta, gamma) values (incl. start)
    logposts : array (n_iter+1,)  log-posterior at each chain state
    accept_rate : float           fraction of proposals accepted
    """
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

    accept_rate = n_accepted / n_iter
    return chain, logposts, accept_rate

"""
Adaptive-tempering SMC for the 5D SEIR posterior.

A dimension-general rewrite of the 2D sampler. The tempering schedule (next_phi),
the ESS calculation and systematic resampling were already dimension-agnostic --
they act on weights and indices, not coordinates. Three things were not, and are
generalised here:

  1. INITIALISATION was hardcoded to (beta_range, gamma_range); it now draws from
     a general box of any dimension.

  2. THE PROPOSAL COVARIANCE used `np.eye(2)` for jitter and no dimensional
     scaling. RWMH proposals must shrink as dimension grows: the optimal scaling
     for a d-dimensional target is 2.38^2/d times the target covariance
     (Roberts & Rosenthal). Without it the acceptance rate collapses -- at d=5
     an unscaled proposal is roughly 2.5x too wide.

  3. THE MOVE STEP called a two-argument likelihood. It now takes a general
     log-likelihood callable, so the same sampler serves SIR, SEIR, or anything
     else with a box prior.

Everything else -- the decoupling of the tempering target from the resampling
threshold (see the 2D methodology), the log-evidence accumulation, the solve
counting -- carries over unchanged.
"""

import numpy as np

from sir_inference.model.seir_score import solve_seir, COMPARTMENT_INDEX, PARAM_NAMES


# ----------------------------------------------------------------------
# Dimension-agnostic helpers (unchanged in substance from the 2D versions)
# ----------------------------------------------------------------------

def effective_sample_size(log_weights):
    """ESS = 1 / sum(W_i^2), computed stably from unnormalised log-weights."""
    m = np.max(log_weights)
    w = np.exp(log_weights - m)
    W = w / w.sum()
    return 1.0 / np.sum(W ** 2)


def systematic_resample(particles, log_weights, rng):
    """Systematic resampling; returns particles, reset weights, and indices."""
    n = particles.shape[0]
    m = np.max(log_weights)
    W = np.exp(log_weights - m)
    W /= W.sum()
    cumulative = np.cumsum(W)
    cumulative[-1] = 1.0
    pointers = rng.uniform(0.0, 1.0 / n) + np.arange(n) / n
    idx = np.clip(np.searchsorted(cumulative, pointers), 0, n - 1)
    return particles[idx].copy(), np.zeros(n), idx


def next_phi(phi_current, loglikes, log_weights, ess_target,
             tol=1e-6, min_step=1e-3):
    """
    Next temperature such that reweighting drops the ESS to `ess_target`.
    Purely arithmetic on stored log-likelihoods -- no ODE solves.
    """
    def ess_at(p):
        return effective_sample_size(log_weights + (p - phi_current) * loglikes)

    if ess_at(1.0) >= ess_target:
        return 1.0
    floor = min(phi_current + min_step, 1.0)
    if ess_at(floor) <= ess_target:
        return floor
    lo, hi = floor, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) > ess_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Dimension-general initialisation and likelihood
# ----------------------------------------------------------------------

def init_particles(n_particles, lo, hi, rng):
    """Draw N particles uniformly from a box of arbitrary dimension. (N, d)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return rng.uniform(lo, hi, size=(n_particles, len(lo)))


def make_seir_loglike(t_obs, data, sigma):
    """
    Build a log-likelihood callable for the SEIR observation model.

    Returns -inf on solver failure rather than raising: extreme parameter values
    do occur in the prior-drawn initial population, and a single failure should
    reject that particle, not abort the run.
    """
    def loglike(theta):
        try:
            u = solve_seir(theta, t_obs, rtol=1e-6, atol=1e-8)
        except Exception:
            return -np.inf
        ll = 0.0
        for name, y_obs in data.items():
            c = COMPARTMENT_INDEX[name]
            ll += -0.5 * np.sum((np.asarray(y_obs) - u[c]) ** 2) / sigma ** 2
        return ll if np.isfinite(ll) else -np.inf
    return loglike


def compute_loglikes(particles, loglike):
    """Log-likelihood at every particle. One ODE solve each -- the dominant cost."""
    return np.array([loglike(p) for p in particles])


# ----------------------------------------------------------------------
# The MOVE step
# ----------------------------------------------------------------------

def move_particles(particles, loglikes, phi, loglike, proposal_cov, n_steps,
                   lo, hi, rng):
    """
    RWMH move targeting the tempered posterior, in any dimension.

    The prior is a uniform box, so the prior term is constant inside it and the
    accept ratio reduces to phi * (loglike_prop - loglike_current). Proposals
    outside the box are rejected WITHOUT an ODE solve -- worth doing explicitly,
    since at high phi the cloud is small relative to the box and out-of-box
    proposals are rare, but early on they are common and each avoided solve is
    a real saving.
    """
    n = particles.shape[0]
    new_p = particles.copy()
    new_ll = loglikes.copy()
    n_accept = 0
    n_solves = 0

    # Cholesky once, then reuse: cheaper than multivariate_normal per proposal.
    try:
        L = np.linalg.cholesky(proposal_cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(proposal_cov + 1e-10 * np.eye(len(lo)))

    for i in range(n):
        theta = new_p[i].copy()
        ll = new_ll[i]
        for _ in range(n_steps):
            prop = theta + L @ rng.standard_normal(len(lo))
            if np.any(prop <= lo) or np.any(prop >= hi):
                continue                      # outside the box: reject, no solve
            ll_prop = loglike(prop)
            n_solves += 1
            if np.log(rng.uniform()) < phi * (ll_prop - ll):
                theta, ll = prop, ll_prop
                n_accept += 1
        new_p[i], new_ll[i] = theta, ll

    return new_p, new_ll, n_accept / (n * n_steps), n_solves


# ----------------------------------------------------------------------
# The sampler
# ----------------------------------------------------------------------

def smc_sampler(loglike, lo, hi, n_particles=2000, ess_frac=0.5,
                resample_frac=0.95, n_move_steps=5, cov_scale=None,
                target_accept=0.25, adapt_rate=1.5,
                rng=None, verbose=True, param_names=None):
    """
    Adaptive-tempering SMC over a uniform box prior of arbitrary dimension.

    Parameters
    ----------
    loglike : callable(theta) -> float
    lo, hi : array-like (d,)     prior box
    cov_scale : float or None    initial proposal covariance multiplier. None uses
                                 the RWMH optimum 2.38^2/d as a starting point.
    target_accept : float        the move step's proposal scale is ADAPTED each
                                 tempering step to hold the acceptance rate near
                                 this value. This is essential in higher dimension:
                                 the tempered target changes shape drastically as
                                 phi climbs from near-uniform to sharply peaked, so
                                 a fixed scale that suits one end fails at the other.
                                 Without adaptation the 5D acceptance rate collapses
                                 below 0.05 and the run stalls.

    Notes on the ESS settings: the tempering TARGET (ess_frac) and the RESAMPLING
    THRESHOLD (resample_frac) are deliberately different -- see the 2D methodology.
    """
    if rng is None:
        rng = np.random.default_rng()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = len(lo)
    N = n_particles
    if cov_scale is None:
        cov_scale = 2.38 ** 2 / d          # starting point; adapted below

    ess_target = ess_frac * N
    ess_threshold = resample_frac * N

    particles = init_particles(N, lo, hi, rng)
    log_weights = np.zeros(N)
    phi = 0.0
    loglikes = compute_loglikes(particles, loglike)
    total_solves = N
    log_evidence = 0.0
    step = 0

    if verbose:
        print(f"SMC in {d}D: {N} particles, ESS target {ess_target:.0f}, "
              f"resample below {ess_threshold:.0f}, {n_move_steps} move steps, "
              f"adaptive scale (target accept {target_accept})")
        print(f"{'step':>4} {'phi':>8} {'d_phi':>8} {'ESS':>7} "
              f"{'accept':>7} {'scale':>7} {'unique':>7} {'solves':>10}")

    while phi < 1.0:
        step += 1
        phi_new = next_phi(phi, loglikes, log_weights, ess_target)
        delta = phi_new - phi

        inc = delta * loglikes
        m = np.max(log_weights)
        W = np.exp(log_weights - m); W /= W.sum()
        log_evidence += np.log(np.sum(W * np.exp(inc - np.max(inc)))) + np.max(inc)

        log_weights = log_weights + inc
        ess = effective_sample_size(log_weights)
        phi = phi_new

        if ess <= ess_threshold:
            particles, log_weights, idx = systematic_resample(particles, log_weights, rng)
            loglikes = loglikes[idx]

        cloud_cov = np.cov(particles.T)
        proposal_cov = cov_scale * cloud_cov + 1e-12 * np.eye(d)
        particles, loglikes, accept, ns = move_particles(
            particles, loglikes, phi, loglike, proposal_cov, n_move_steps,
            lo, hi, rng)
        total_solves += ns

        # Adapt the proposal scale toward the target acceptance rate. The move
        # multiplies the scale by exp(adapt_rate * (accept - target)): too many
        # rejections shrink it, too many acceptances grow it.
        cov_scale *= np.exp(adapt_rate * (accept - target_accept))

        if verbose:
            print(f"{step:>4} {phi:>8.4f} {delta:>8.4f} {ess:>7.1f} {accept:>7.3f} "
                  f"{cov_scale:>7.3f} {len(np.unique(particles, axis=0)):>7d} "
                  f"{total_solves:>10,}")

    if verbose:
        names = param_names or [f"p{j}" for j in range(d)]
        print(f"\nDone: phi=1 in {step} steps, {total_solves:,} ODE solves.")
        for j, nm in enumerate(names):
            print(f"  {nm:9s} = {particles[:, j].mean():.4f} +- {particles[:, j].std():.4f}")

    return dict(particles=particles, log_weights=log_weights, n_steps=step,
                total_solves=total_solves, log_evidence=log_evidence)


if __name__ == "__main__":
    from sir_inference.model.seir_score import default_prior

    d5 = np.load("seir_data.npz")
    t_obs = d5["t_obs"]; sigma = float(d5["sigma"])
    theta_true = d5["theta_true"]
    observed = [str(c) for c in d5["observed"]]
    data = {c: d5[f"data_{c}"] for c in observed}

    prior = default_prior()
    loglike = make_seir_loglike(t_obs, data, sigma)

    print(f"Data: {observed} observed, sigma={sigma}, "
          f"{len(t_obs)} times over [{t_obs[0]:.0f},{t_obs[-1]:.0f}]d")
    print(f"Truth: " + ", ".join(f"{n}={v}" for n, v in zip(PARAM_NAMES, theta_true)))
    print()

    rng = np.random.default_rng(0)
    res = smc_sampler(loglike, prior.lo, prior.hi, n_particles=2000,
                      n_move_steps=5, rng=rng, param_names=PARAM_NAMES)

    p = res["particles"]
    print(f"\n  truth      = {np.round(theta_true, 4)}")
    R0 = p[:, 0] / np.clip(p[:, 2], 1e-9, None)
    print(f"  R0         = {R0.mean():.3f} +- {R0.std():.3f}   (truth 3.0)")

    np.savez("smc_run_5d.npz", particles=p, log_weights=res["log_weights"],
             total_solves=res["total_solves"], t_obs=t_obs, sigma=sigma,
             theta_true=theta_true, observed=d5["observed"],
             **{f"data_{c}": data[c] for c in observed})
    print(f"\nSaved -> smc_run_5d.npz  ({res['total_solves']:,} solves)")
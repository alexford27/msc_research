"""
The GRIDDED TRUE POSTERIOR -- an independent reference for evaluating all methods.

WHY THIS EXISTS
---------------
In the four-cell grid we must measure how far each cloud (raw HM, thinned HM,
raw SMC, thinned SMC) sits from the true posterior pi. Using the SMC particles
as that reference is circular: SMC then scores zero BY CONSTRUCTION, and the
thinned-SMC cell is flattered because it is a subset of the reference itself.

At 2D -- and only at 2D -- we can sidestep this entirely by evaluating the
unnormalised posterior on a dense grid and treating that as ground truth. This is
exactly why 2D is the validation tier of the study: it is the one dimension where
the answer is knowable independently of any inference method, so every method can
be scored against something none of them produced.

The grid is NOT an inference method being compared; it is the ruler.

COST NOTE
---------
The grid costs n_grid^2 ODE solves (e.g. 200x200 = 40,000), which is MORE than
SMC. That is fine: it is a validation instrument, not a competitor, and its cost
is never counted in the method comparison. It is infeasible beyond ~3 dimensions,
which is precisely why the higher-dimensional tiers must fall back to comparing
methods against each other.

OUTPUT
------
A weighted grid (for exact expectations) and an i.i.d. reference sample drawn
from it (for sample-based metrics such as the energy distance).
"""

import numpy as np

from sir_inference.model.sir_score import solve_sir_with_sensitivities, COMPARTMENT_INDEX, default_prior


def log_posterior_grid(t_obs, data, sigma, prior,
                       beta_range=(0.30, 1.10), gamma_range=(0.08, 0.40),
                       n_grid=160, y0=(0.99, 0.01, 0.0),
                       rtol=1e-6, atol=1e-8, verbose=True):
    """
    Evaluate the unnormalised log-posterior on a dense grid.

    The default box is deliberately TIGHTER than the prior box (beta in [0,2],
    gamma in [0,1]): posterior mass is concentrated in a small sub-region, so
    gridding the full prior box would waste almost every solve on parameter values
    with negligible density. The box is checked below (see `mass_within_box`) to
    confirm it captures essentially all the mass -- if it does not, widen it.

    Returns
    -------
    betas, gammas : 1-D grids
    logpost       : (n_gamma, n_beta) unnormalised log-posterior
    n_solves      : int, ODE solves used (validation cost, not a method cost)
    """
    betas = np.linspace(*beta_range, n_grid)
    gammas = np.linspace(*gamma_range, n_grid)
    logpost = np.full((n_grid, n_grid), -np.inf)

    n_solves = 0
    for a, g in enumerate(gammas):
        if verbose and a % max(1, n_grid // 10) == 0:
            print(f"  grid row {a}/{n_grid}")
        for b, be in enumerate(betas):
            try:
                u, _ = solve_sir_with_sensitivities(be, g, t_obs, y0=y0,
                                                    rtol=rtol, atol=atol)
                n_solves += 1
                ll = 0.0
                for name, y_obs in data.items():
                    c = COMPARTMENT_INDEX[name]
                    ll += -0.5 * np.sum((np.asarray(y_obs) - u[c]) ** 2) / sigma ** 2
                logpost[a, b] = ll + prior.logpdf(np.array([be, g]))
            except Exception:
                pass

    return betas, gammas, logpost, n_solves


def grid_to_weights(logpost):
    """Normalise a log-posterior grid to probability weights (log-sum-exp stable)."""
    w = np.exp(logpost - np.nanmax(logpost))
    w[~np.isfinite(w)] = 0.0
    return w / w.sum()


def sample_from_grid(betas, gammas, weights, n_samples=5000, rng=None, jitter=True):
    """
    Draw an i.i.d. reference sample from the gridded posterior.

    Sample-based metrics (energy distance) need points, not a density, so we
    convert the grid into a cloud by sampling cells with probability equal to
    their weight. `jitter` spreads each draw uniformly within its cell so the
    reference is a genuine continuous sample rather than a lattice -- without it,
    the discreteness of the grid would itself register as a distance.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n_g, n_b = weights.shape
    flat = weights.ravel()
    idx = rng.choice(flat.size, size=n_samples, replace=True, p=flat)
    ga, be = np.unravel_index(idx, (n_g, n_b))
    pts = np.column_stack([betas[be], gammas[ga]])
    if jitter:
        db = betas[1] - betas[0]
        dg = gammas[1] - gammas[0]
        pts[:, 0] += rng.uniform(-db / 2, db / 2, n_samples)
        pts[:, 1] += rng.uniform(-dg / 2, dg / 2, n_samples)
    return pts


def grid_moments(betas, gammas, weights):
    """Exact posterior moments from the grid (no Monte Carlo error)."""
    B, G = np.meshgrid(betas, gammas)
    mb = (weights * B).sum(); mg = (weights * G).sum()
    sb = np.sqrt((weights * (B - mb) ** 2).sum())
    sg = np.sqrt((weights * (G - mg) ** 2).sum())
    R = B / np.clip(G, 1e-12, None)
    mR = (weights * R).sum()
    sR = np.sqrt((weights * (R - mR) ** 2).sum())
    return dict(beta_mean=mb, beta_sd=sb, gamma_mean=mg, gamma_sd=sg,
                R0_mean=mR, R0_sd=sR)


def mass_within_box(betas, gammas, weights, edge_frac=0.02):
    """
    Fraction of posterior mass sitting within `edge_frac` of the grid boundary.

    A large value means the grid box is CLIPPING the posterior, so the reference
    would be truncated and every distance measured against it biased. Should be
    negligible; if not, widen the box and recompute.
    """
    n_g, n_b = weights.shape
    eb = max(1, int(edge_frac * n_b)); eg = max(1, int(edge_frac * n_g))
    mask = np.zeros_like(weights, dtype=bool)
    mask[:eg, :] = mask[-eg:, :] = True
    mask[:, :eb] = mask[:, -eb:] = True
    return float(weights[mask].sum())


def build_reference(smc_file="smc_run.npz", n_grid=110, n_samples=5000,
                    beta_range=(0.30, 1.10), gamma_range=(0.08, 0.40),
                    out_file="grid_reference.npz", verbose=True):
    """Build the gridded truth from the SAME dataset the methods were run on."""
    smc = np.load(smc_file)
    t_obs = smc["t_obs"]
    sigma = float(smc["sigma"])
    data = {c: smc[f"data_{c}"] for c in ("s", "i", "r") if f"data_{c}" in smc}
    prior = default_prior()

    if verbose:
        print(f"Gridding the true posterior: {n_grid}x{n_grid} "
              f"= {n_grid**2:,} ODE solves (validation cost, not a method cost)")

    betas, gammas, logpost, n_solves = log_posterior_grid(
        t_obs, data, sigma, prior, beta_range, gamma_range, n_grid, verbose=verbose)
    weights = grid_to_weights(logpost)

    edge = mass_within_box(betas, gammas, weights)
    mom = grid_moments(betas, gammas, weights)
    ref = sample_from_grid(betas, gammas, weights, n_samples)

    if verbose:
        print(f"\nGrid box: beta {beta_range}, gamma {gamma_range}")
        print(f"  mass within 2% of the box edge: {edge:.2e}"
              f"   -> {'SAFE (box captures the posterior)' if edge < 1e-4 else 'WARNING: widen the box'}")
        print(f"\nEXACT posterior moments from the grid:")
        print(f"  beta  = {mom['beta_mean']:.4f} +- {mom['beta_sd']:.4f}")
        print(f"  gamma = {mom['gamma_mean']:.4f} +- {mom['gamma_sd']:.4f}")
        print(f"  R0    = {mom['R0_mean']:.4f} +- {mom['R0_sd']:.4f}   (truth 3.0)")
        print(f"\nReference sample: {ref.shape}")

    np.savez(out_file, betas=betas, gammas=gammas, logpost=logpost,
             weights=weights, reference=ref, n_solves=n_solves,
             **{f"moment_{k}": v for k, v in mom.items()})
    if verbose:
        print(f"Saved -> {out_file}")
    return ref, mom


if __name__ == "__main__":
    build_reference()
"""
The FOUR-CELL GRID: Marina's core experiment.

Four objects, all measured by their discrepancy to the true posterior pi:

                    |  raw                    |  thinned toward pi
    ----------------|-------------------------|--------------------------
    HM  (approx)    |  non-implausible set    |  thinned-HM
    SMC (exact)     |  posterior particles    |  thinned-SMC

Expected orderings (Marina's hypotheses):
    raw HM   >>  raw SMC          (approximate arm starts far behind)
    raw HM   >>  thinned HM       (thinning rescues the set)
    raw SMC   ~  thinned SMC      (thinning is ~free on an already-good sample)
    thinned HM  ?  thinned SMC    <- THE OPEN QUESTION (the central finding)

The "?" isolates the SET-FORMATION error: both thinned sets use the SAME target
(true pi) and the SAME exact score (forward sensitivities), so the only remaining
difference is WHICH CANDIDATES each arm supplies. Any residual gap is therefore
attributable to the quality of the HM candidate set, not to the score.

METRIC CHOICE -- avoiding circularity:
Stein thinning SELECTS points by minimising KSD, so scoring the result by KSD
would be partly self-fulfilling (thinned sets look good by construction). We
therefore report an INDEPENDENT metric alongside:
  - energy distance to a reference posterior sample (impartial, not the thinning
    objective), plus simple moment/coverage summaries.
KSD is still reported, but read as "the objective that was optimised", not as
neutral evidence.

Inputs: smc_run.npz (SMC particles + dataset) and hm_run.npz (HM candidates).
Scores come from sir_score.posterior_scores (exact, via forward sensitivities).
"""

import numpy as np

from sir_inference.model.sir_score import posterior_scores, default_prior, COMPARTMENT_INDEX


# ----------------------------------------------------------------------
# INDEPENDENT EVALUATION METRIC: energy distance
# ----------------------------------------------------------------------

def energy_distance(X, Y, max_n=2000, rng=None):
    """
    Energy distance between two point clouds:

        E(X, Y) = 2 E|X - Y|  -  E|X - X'|  -  E|Y - Y'|

    Zero iff the two distributions coincide; strictly positive otherwise.

    Why this metric: it is NOT the quantity Stein thinning optimises, so it gives
    an impartial reading of how well each cloud represents the target. (Scoring
    the thinned sets by KSD alone would be circular.) It is also distribution-free
    and needs no density -- only samples.

    Subsampled to max_n points per cloud to keep the pairwise distances tractable.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    def mean_dist(A, B):
        # mean pairwise Euclidean distance between two clouds
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return d.mean()

    return float(2 * mean_dist(X, Y) - mean_dist(X, X) - mean_dist(Y, Y))


def standardised_energy_distance(X, Y, scale, **kw):
    """
    Energy distance computed after dividing both clouds by `scale` (per-dimension).

    Why: beta and gamma have different natural scales, so a raw Euclidean metric
    weights them arbitrarily. Standardising by the reference posterior's own
    standard deviations expresses the distance in 'posterior standard deviations',
    which is interpretable and dimension-fair.
    """
    return energy_distance(np.asarray(X) / scale, np.asarray(Y) / scale, **kw)


# ----------------------------------------------------------------------
# THE OBJECTIVE THAT IS OPTIMISED: KSD (reported, but not neutral)
# ----------------------------------------------------------------------

def imq_ksd(sample, gradient, c=1.0, beta=-0.5):
    """
    Kernel Stein discrepancy with an inverse-multiquadric base kernel,
    computed as the V-statistic  KSD^2 = (1/n^2) sum_ij k_p(x_i, x_j).

    Reported for completeness, but note this is the very quantity Stein thinning
    minimises -- so it is NOT an impartial score for the thinned sets. Use the
    energy distance for the fair comparison.

    IMQ:  k(x,y) = (c^2 + |x-y|^2)^beta,  beta in (-1, 0).
    Stein kernel (Langevin operator applied in both arguments) for this kernel has
    the standard closed form used below.
    """
    x = np.atleast_2d(sample); g = np.atleast_2d(gradient)
    # The V-statistic needs an n x n x d array; subsample large clouds so this
    # stays in memory (KSD is an average, so a subsample estimates it fine).
    max_n = 1500
    if len(x) > max_n:
        sub = np.random.default_rng(0).choice(len(x), max_n, replace=False)
        x, g = x[sub], g[sub]
    n, d = x.shape
    diff = x[:, None, :] - x[None, :, :]          # (n,n,d)
    sq = (diff ** 2).sum(-1)                       # (n,n)
    base = c ** 2 + sq

    # score-score term
    gg = g @ g.T                                   # (n,n)
    t1 = gg * base ** beta

    # cross terms (score . grad of kernel)
    gdiff = np.einsum('ik,ijk->ij', g, diff)       # g_i . (x_i - x_j)
    gdiff_j = np.einsum('jk,ijk->ij', g, diff)     # g_j . (x_i - x_j)
    t2 = -2 * beta * (gdiff - gdiff_j) * base ** (beta - 1)

    # kernel-derivative term
    t3 = (-2 * beta * d * base ** (beta - 1)
          - 4 * beta * (beta - 1) * sq * base ** (beta - 2))

    kp = t1 + t2 + t3
    return float(np.sqrt(max(kp.sum() / n ** 2, 0.0)))


# ----------------------------------------------------------------------
# SUMMARY OF A CLOUD
# ----------------------------------------------------------------------

def summarise(points, label, reference=None, ref_scale=None, gradient=None):
    """Moments, R0, and (if a reference is given) distance to it."""
    pts = np.atleast_2d(points)
    b, g = pts[:, 0], pts[:, 1]
    R0 = b / np.clip(g, 1e-12, None)
    row = {
        "label": label, "n": len(pts),
        "beta_mean": b.mean(), "beta_sd": b.std(),
        "gamma_mean": g.mean(), "gamma_sd": g.std(),
        "R0_mean": R0.mean(), "R0_sd": R0.std(),
    }
    if reference is not None:
        row["energy"] = standardised_energy_distance(pts, reference, ref_scale)
    if gradient is not None:
        row["ksd"] = imq_ksd(pts, gradient)
    return row


# ----------------------------------------------------------------------
# THE GRID
# ----------------------------------------------------------------------

def run_grid(smc_file="smc_run.npz", hm_file="hm_run.npz",
             m_thin=200, verbose=True):
    """
    Build and evaluate all four cells.

    m_thin : how many points each thinned set retains.
    """
    smc = np.load(smc_file)
    hm = np.load(hm_file)

    smc_pts = smc["particles"]
    hm_pts = hm["samples"]
    t_obs = smc["t_obs"]
    sigma = float(smc["sigma"])
    data = {c: smc[f"data_{c}"] for c in ("s", "i", "r") if f"data_{c}" in smc}
    prior = default_prior()

    # The REFERENCE for the impartial metric is the SMC posterior itself
    # (our best available proxy for pi). Distances are expressed in units of
    # its own per-parameter standard deviations.
    reference = smc_pts
    ref_scale = smc_pts.std(axis=0)

    if verbose:
        print("=" * 78)
        print("FOUR-CELL GRID")
        print(f"  SMC : {len(smc_pts)} particles, {int(smc['total_solves']):,} ODE solves")
        print(f"  HM  : {len(hm_pts)} candidates, {int(hm['total_solves']):,} ODE solves")
        print(f"  thinning to m = {m_thin} points, exact score via forward sensitivities")
        print("=" * 78)

    # --- exact scores at every candidate (forward sensitivities) ---
    if verbose:
        print("\nComputing exact scores...")
    smc_scores = posterior_scores(smc_pts, t_obs, data, sigma, prior, n_jobs=-1, verbose=True)
    hm_scores = posterior_scores(hm_pts, t_obs, data, sigma, prior, n_jobs=-1, verbose=True)
    if verbose:
        print(f"  SMC scores {smc_scores.shape}, HM scores {hm_scores.shape}")

    # --- thin both toward the TRUE posterior using those scores ---
    from stein_thinning.thinning import thin
    if verbose:
        print(f"\nStein thinning (IMQ kernel, m={m_thin})...")
    idx_smc = thin(smc_pts, smc_scores, m_thin)
    idx_hm = thin(hm_pts, hm_scores, m_thin)
    smc_thin = smc_pts[idx_smc]
    hm_thin = hm_pts[idx_hm]

    # --- evaluate all four cells ---
    rows = [
        summarise(hm_pts,   "HM (raw)",        reference, ref_scale, hm_scores),
        summarise(hm_thin,  "HM (thinned)",    reference, ref_scale, hm_scores[idx_hm]),
        summarise(smc_pts,  "SMC (raw)",       reference, ref_scale, smc_scores),
        summarise(smc_thin, "SMC (thinned)",   reference, ref_scale, smc_scores[idx_smc]),
    ]
    return rows, dict(smc_thin=smc_thin, hm_thin=hm_thin,
                      idx_smc=idx_smc, idx_hm=idx_hm,
                      smc_scores=smc_scores, hm_scores=hm_scores,
                      reference=reference, ref_scale=ref_scale)


def print_table(rows):
    print("\n" + "=" * 78)
    print("RESULTS  (energy distance to the SMC posterior, in posterior-sd units)")
    print("=" * 78)
    print(f"{'cell':16s} {'n':>6s} {'beta':>14s} {'gamma':>14s} {'R0':>14s} "
          f"{'energy':>9s} {'KSD':>9s}")
    for r in rows:
        print(f"{r['label']:16s} {r['n']:>6d} "
              f"{r['beta_mean']:.3f}+-{r['beta_sd']:.3f} "
              f"{r['gamma_mean']:.3f}+-{r['gamma_sd']:.3f} "
              f"{r['R0_mean']:.2f}+-{r['R0_sd']:.2f} "
              f"{r.get('energy', float('nan')):9.4f} {r.get('ksd', float('nan')):9.4f}")
    print("\n  truth: beta=0.6, gamma=0.2, R0=3.0")
    print("  energy: lower = closer to the reference posterior (IMPARTIAL metric)")
    print("  KSD   : the quantity thinning optimises -- NOT impartial for thinned rows")


if __name__ == "__main__":
    import sys
    hm_file = sys.argv[1] if len(sys.argv) > 1 else "hm_run.npz"
    rows, extra = run_grid(hm_file=hm_file, m_thin=200)
    print_table(rows)

    # --- The four comparisons Marina drew ---
    e = {r["label"]: r["energy"] for r in rows}
    print("\n" + "=" * 78)
    print("MARINA'S FOUR COMPARISONS (energy distance, impartial)")
    print("=" * 78)
    print(f"  raw HM vs raw SMC        : {e['HM (raw)']:.4f} vs {e['SMC (raw)']:.4f}"
          f"   -> HM {'>>' if e['HM (raw)'] > 3*e['SMC (raw)'] else '>'} SMC")
    print(f"  raw HM vs thinned HM     : {e['HM (raw)']:.4f} vs {e['HM (thinned)']:.4f}"
          f"   -> thinning {'rescues' if e['HM (thinned)'] < e['HM (raw)'] else 'does NOT help'}")
    print(f"  raw SMC vs thinned SMC   : {e['SMC (raw)']:.4f} vs {e['SMC (thinned)']:.4f}"
          f"   -> {'~equal' if abs(e['SMC (thinned)']-e['SMC (raw)']) < 0.5*e['SMC (raw)']+1e-9 else 'changed'}")
    print(f"  thinned HM vs thinned SMC: {e['HM (thinned)']:.4f} vs {e['SMC (thinned)']:.4f}"
          f"   <- THE '?'")
    gap = e['HM (thinned)'] - e['SMC (thinned)']
    print(f"\n  RESIDUAL GAP = {gap:.4f}  (the set-formation error:")
    print( "     same target, same exact score, same algorithm -- only the CANDIDATES differ)")
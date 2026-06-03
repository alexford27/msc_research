"""
Numba-accelerated SIR forward model (drop-in replacement for solve_sir).

WHY: profiling showed ~123 ODE solves/sec, dominated by the per-step Python
function-call overhead (solve_ivp calls the RHS thousands of times per solve,
each call crossing the Python/C boundary). JIT-compiling the RHS arithmetic with
numba removes almost all that overhead.

DESIGN:
  - The pure-arithmetic SIR rates are JIT-compiled with @njit (the hot path).
  - solve_sir keeps an IDENTICAL interface to the original, so log_likelihood and
    rwmh need NO changes -- they just run faster underneath.
  - GRACEFUL FALLBACK: if numba isn't installed (or fails), we fall back to a plain
    Python RHS so nothing breaks; you'll just lose the speedup.
  - The first call pays a one-off compilation cost (~1s); all later calls are fast.

NOTE: scipy's solve_ivp still calls the RHS from Python, so this removes the
*arithmetic* overhead but not 100% of the call overhead. For SIR (tiny RHS), the
arithmetic/boundary cost is the dominant share, so the speedup is typically large.
If you later need more, the next rungs are parallelising particle solves (for SMC)
and, only as a last resort, a C-based solver (SUNDIALS/CVODE) or diffrax/JAX.
"""

import numpy as np
from scipy.integrate import solve_ivp

# --- Try to import numba; fall back gracefully if unavailable ---
try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    # Define a no-op decorator so the code still runs without numba.
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        # support both @njit and @njit(...) usage
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrap


# ----------------------------------------------------------------------
# THE HOT PATH: pure SIR rate arithmetic, JIT-compiled
# ----------------------------------------------------------------------

@njit(cache=True)
def _sir_rates(s, i, r, beta, gamma):
    """
    Compiled SIR rates in fractions. Returns (ds, di, dr).
    This tiny function is called thousands of times per solve, so compiling it
    is where the speedup comes from.
    """
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return ds, di, dr


# ----------------------------------------------------------------------
# solve_sir: SAME interface as before, faster underneath
# ----------------------------------------------------------------------

def solve_sir(beta, gamma, t_eval, y0=(0.99, 0.01, 0.0),
              t_span=None, rtol=1e-5, atol=1e-7):
    """
    Solve the SIR system forward and return the trajectory at t_eval.
    Interface identical to the original solve_sir; only the RHS is accelerated.
    """
    y0 = np.asarray(y0, dtype=float)
    if not np.isclose(y0.sum(), 1.0):
        raise ValueError(f"Initial conditions must sum to 1, got {y0.sum()}")

    t_eval = np.asarray(t_eval, dtype=float)
    if t_span is None:
        t_span = (t_eval[0], t_eval[-1])

    # The RHS solve_ivp calls. We close over (beta, gamma) rather than using
    # args=, which keeps the call path simple and numba-friendly.
    def rhs(t, y):
        ds, di, dr = _sir_rates(y[0], y[1], y[2], beta, gamma)
        return (ds, di, dr)

    sol = solve_ivp(
        fun=rhs,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y


# ----------------------------------------------------------------------
# BENCHMARK  (run this file directly to measure the speedup)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print(f"numba available: {_HAVE_NUMBA}")

    BETA, GAMMA = 0.6, 0.2
    T_OBS = np.linspace(0, 60, 30)
    Y0 = (0.99, 0.01, 0.0)

    # --- Warm-up: trigger JIT compilation (first call compiles; don't time it) ---
    print("Warming up (compiling)...")
    _ = solve_sir(BETA, GAMMA, T_OBS, y0=Y0)

    # --- Time many solves (this is what the samplers actually do) ---
    n_solves = 2000
    print(f"Timing {n_solves} solves...")
    start = time.time()
    for _ in range(n_solves):
        # vary params slightly so it's realistic (and not trivially cached)
        b = BETA + np.random.uniform(-0.05, 0.05)
        g = GAMMA + np.random.uniform(-0.02, 0.02)
        _ = solve_sir(b, g, T_OBS, y0=Y0)
    elapsed = time.time() - start

    rate = n_solves / elapsed
    print(f"\n{n_solves} solves in {elapsed:.2f}s")
    print(f"--> {rate:.0f} solves/sec  ({1000/rate:.2f} ms per solve)")
    print(f"\nCompare to your earlier ~123 solves/sec to see the speedup.")
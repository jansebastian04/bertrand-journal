"""
mstar_delta_grid.py
====================
For a grid of (s, l, n) game parameters, compute two companion quantities
tied to the flat-block construction behind Theorem no-minty(ii):

  1. m^star(n,l,s): the critical projection resolution from the theorem's
     own closed-form criterion (eq:mstar-app / eq:grid-criterion),

         A_k(t) = (1 - (1-t)^(nu+k-1)) / (nu+k-1),   t = 1/m,  nu = l*n
         rho_m  = A_0(t) A_2(t) / A_1(t)^2
         m^star = min{ m0 >= 2 : rho_m < (1+r)^2/(4r) for every m >= m0 }

     -- purely closed-form, independent of delta and of the actual game
     gradient. rho_m is monotone decreasing to 1 in every instance checked
     (target (1+r)^2/(4r) is always > 1), so a plain backward scan
     correctly finds this threshold.

  2. delta_max(n,l,s): at m = m^star, the flat-block witness
     x(delta) = (delta, kappa,...,kappa), kappa = (m*w_dagger - delta)/(m-1),
     w_dagger = (1+r)*A_1(1/m) / (2*A_0(1/m)),
     is fed into the ACTUAL (raw, unweighted) game gradient
     Bertrand(s,l,n,m).gradient -- see src/bertrand.py -- and delta is
     pushed up from 0 until (x(delta)-x*)^T v(x(delta)) first turns
     non-positive. delta_max is the largest lower-slope bound for which
     THIS SPECIFIC witness still exhibits a Minty violation at this
     instance's own m^star -- not a claim about every witness or every m,
     just this one theorem-prescribed construction at m^star.

Background
----------
Theorem no-minty(ii), as stated, only promises failure "for sufficiently
small delta" once m >= m^star; delta_max quantifies "sufficiently small"
for the flat-block witness specifically. Every (s,l,n) combination checked
so far (s in 0..3, l in 1..3, n in 2..4) shows a single, clean sign
crossing (confirmed by a full scan before bisecting), so delta_max is
well-defined by bisection throughout; find_delta_max flags
MULTI_CROSSING / boundary anomalies if that ever fails to hold for a new
combination.

Usage
-----
    python scripts/geometry/mstar_delta_grid.py
    python scripts/geometry/mstar_delta_grid.py --s 0 1 2 3 --l 1 2 3 --n 2 3 4
    python scripts/geometry/mstar_delta_grid.py --out results/mstar_delta_grid.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from bertrand import Bertrand  # local src/bertrand.py

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "mstar_delta_grid.csv")


# ===========================================================================
# m^star from the theorem's own closed-form criterion (eq:mstar-app)
# ===========================================================================

def A(k, t, nu):
    return (1 - (1 - t) ** (nu + k - 1)) / (nu + k - 1)


def rho(m, nu):
    t = 1.0 / m
    return A(0, t, nu) * A(2, t, nu) / A(1, t, nu) ** 2


def criterion_rhs(alpha):
    return 1 + 1.0 / (4 * alpha * (alpha + 1))


def compute_mstar(alpha, nu, m_max=200):
    """min{m0 >= 2 : rho_m < criterion_rhs(alpha) for every m >= m0}."""
    rhs = criterion_rhs(alpha)
    holds = [rho(m, nu) < rhs for m in range(2, m_max + 1)]
    for i in range(len(holds) - 1, -1, -1):
        if not holds[i]:
            return (i + 1) + 2
    return 2


def w_dagger(m, r, nu):
    t = 1.0 / m
    return (1 + r) * A(1, t, nu) / (2 * A(0, t, nu))


# ===========================================================================
# delta_max: push the flat-block witness's delta up against the ACTUAL raw
# game gradient until the Minty violation it exhibits disappears
# ===========================================================================

def witness(delta, m, wdag):
    kappa = (m * wdag - delta) / (m - 1)
    x = np.full(m, kappa)
    x[0] = delta
    return x


def minty_val(game, xstar, m, wdag, delta):
    x = witness(delta, m, wdag)
    v = game.gradient(x)
    return float(np.dot(x - xstar, v))


def find_delta_max(game, xstar, m, wdag, n_scan=500):
    """Largest delta in (0, w_dagger) -- w_dagger is the feasibility bound,
    kappa(delta) >= delta -- for which minty_val(delta) > 0. Scans first to
    find the sign pattern (and flag anomalies), then bisects the crossing."""
    hi_bound = wdag
    deltas = np.linspace(hi_bound * 1e-4, hi_bound * (1 - 1e-6), n_scan)
    vals = np.array([minty_val(game, xstar, m, wdag, d) for d in deltas])
    pos = vals > 0

    if not pos[0]:
        return 0.0, vals[0], "NEGATIVE_AT_SMALLEST_SCANNED_DELTA"
    if pos.all():
        return deltas[-1], vals[-1], "STAYS_POSITIVE_TO_FEASIBILITY_BOUND"

    idx = np.argmax(~pos)
    n_sign_changes = np.sum(np.diff(pos.astype(int)) != 0)
    flag = "" if n_sign_changes == 1 else f"MULTI_CROSSING({n_sign_changes})"

    lo, hi = deltas[idx - 1], deltas[idx]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if minty_val(game, xstar, m, wdag, mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo, minty_val(game, xstar, m, wdag, lo), flag


# ===========================================================================
# Grid sweep
# ===========================================================================

CSV_FIELDS = ["s", "l", "n", "alpha", "mstar", "r", "w_dagger", "delta_max", "note"]


def run_grid(s_list, l_list, n_list, out_path, verbose=True):
    rows = []
    header = (f"{'s':>2}{'l':>3}{'n':>3}{'alpha':>7}{'m*':>4}{'r':>8}"
              f"{'w_dag':>9}{'delta_max':>12}{'note':>28}")
    if verbose:
        print(header)
    for s in s_list:
        for l in l_list:
            for n in n_list:
                a = l * (n - 1)
                alpha = a + s
                r = alpha / (alpha + 1)
                nu = l * n
                mstar = compute_mstar(alpha, nu)
                wdag = w_dagger(mstar, r, nu)

                game = Bertrand(s=s, l=l, n=n, m=mstar)
                xstar = game.bne
                delta_max, val, note = find_delta_max(game, xstar, mstar, wdag)

                rows.append(dict(s=s, l=l, n=n, alpha=alpha, mstar=mstar,
                                  r=r, w_dagger=wdag, delta_max=delta_max, note=note))
                if verbose:
                    print(f"{s:>2}{l:>3}{n:>3}{alpha:>7}{mstar:>4}{r:>8.4f}{wdag:>9.4f}"
                          f"{delta_max:>12.6f}{note:>28}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    if verbose:
        print(f"\nWrote {len(rows)} rows to {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--s", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--l", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    run_grid(args.s, args.l, args.n, args.out)


if __name__ == "__main__":
    main()

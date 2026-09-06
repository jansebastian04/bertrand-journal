"""
mstar_tightness_check.py
=========================
Is m^star (results/mstar_delta_grid.csv) a TIGHT threshold, or just a
sufficient one? For every (s, l, n) row of that grid, and every m in
2..mstar (inclusive), this checks whether the RAW game instance
Bertrand(s,l,n,m) satisfies the Minty condition

    (x - x*)^T v(x)  <=  0   for every x in B_delta^m
                                = {x : x_i >= delta, sum_k x_k <= m}

at delta = that row's own delta_max (the flat-block witness's crossing
value, computed AT m = mstar by mstar_delta_grid.py -- reused here
unchanged across the whole m-sweep for that row).

If m* is tight, every m < mstar should come back "satisfies" (no witness
anywhere in the box beats x*), and only m = mstar (and above) should
"VIOLATE". A "VIOLATES" verdict at some m < mstar would mean the
closed-form m* criterion is not necessary -- Minty can already fail
below it.

Search combines sampling, multi-seed differential evolution, SLSQP polishing,
and a fine grid. The label "satisfies" means no violating witness was found;
it is not a proof of the universal inequality. Default seed: 0.

Usage
-----
    python scripts/geometry/mstar_tightness_check.py
    python scripts/geometry/mstar_tightness_check.py --csv results/mstar_delta_grid.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution, minimize

_SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from bertrand import Bertrand  # local src/bertrand.py

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DEFAULT_CSV = os.path.join(RESULTS_DIR, "mstar_delta_grid.csv")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "mstar_tightness_check.csv")


# ===========================================================================
# Geometry of B_delta^m = {x_i >= delta, sum_k x_k <= m}.
# ===========================================================================

def sample_simplex_interior(rng, m, delta, n_points):
    R = m * (1.0 - delta)
    d = rng.dirichlet(np.ones(m + 1), size=n_points)
    return delta + R * d[:, :m]


def sample_face_lower(rng, m, delta, face_i, n_points):
    R = m * (1.0 - delta)
    d = rng.dirichlet(np.ones(m), size=n_points)
    free = delta + R * d[:, :m - 1]
    X = np.empty((n_points, m))
    other = [j for j in range(m) if j != face_i]
    X[:, other] = free
    X[:, face_i] = delta
    return X


def sample_face_budget(rng, m, delta, n_points):
    R = m * (1.0 - delta)
    d = rng.dirichlet(np.ones(m), size=n_points)
    free = delta + R * d[:, :m - 1]
    last = m - free.sum(axis=1)
    return np.column_stack([free, last])


def exact_vertices(m, delta):
    tall = m - (m - 1) * delta
    V = np.full((m + 1, m), delta)
    for i in range(m):
        V[i + 1, i] = tall
    return V


def fine_grid(m, delta, grid_n):
    xs = np.linspace(delta, m - delta, grid_n)
    mesh = np.meshgrid(*([xs] * m), indexing="ij")
    pts = np.stack([g.ravel() for g in mesh], axis=-1)
    mask = pts.sum(axis=1) <= m + 1e-12
    return pts[mask]


# ===========================================================================
# Minty objective and numerical maximization using multiple search methods.
# ===========================================================================

def minty_values(game, x_star, X):
    X = np.atleast_2d(X)
    V = game.gradient(X)
    return np.sum((X - x_star) * V, axis=-1)


def robust_max_minty(game, x_star, m, delta, rng, n_interior, n_face,
                      n_de_seeds, grid_n, verbose=False):
    candidates = [(0.0, x_star.copy())]

    interior = sample_simplex_interior(rng, m, delta, n_interior)
    vals = minty_values(game, x_star, interior)
    candidates.append((vals.max(), interior[vals.argmax()]))

    for i in range(m):
        face_pts = sample_face_lower(rng, m, delta, i, n_face)
        vals = minty_values(game, x_star, face_pts)
        candidates.append((vals.max(), face_pts[vals.argmax()]))

    budget_pts = sample_face_budget(rng, m, delta, n_face)
    vals = minty_values(game, x_star, budget_pts)
    candidates.append((vals.max(), budget_pts[vals.argmax()]))

    V = exact_vertices(m, delta)
    vals = minty_values(game, x_star, V)
    candidates.append((vals.max(), V[vals.argmax()]))

    if grid_n:
        grid_pts = fine_grid(m, delta, grid_n)
        vals = minty_values(game, x_star, grid_pts)
        candidates.append((vals.max(), grid_pts[vals.argmax()]))

    def neg_obj(xv):
        x = np.asarray(xv)
        excess = max(0.0, x.sum() - m)
        return -minty_values(game, x_star, x)[0] + 1e3 * excess

    bounds = [(delta, m - delta)] * m
    for seed in range(n_de_seeds):
        res = differential_evolution(neg_obj, bounds, seed=seed, tol=1e-15,
                                      maxiter=400, popsize=30, polish=False)
        candidates.append((-res.fun, res.x))

    cons = ({"type": "ineq", "fun": lambda x: m - np.sum(x)},)
    slsqp_bounds = [(delta, None)] * m

    def neg_val(xv):
        return -minty_values(game, x_star, np.asarray(xv))[0]

    best_val, best_x = -np.inf, None
    for val, x0 in candidates:
        x0 = np.clip(np.asarray(x0, dtype=float), delta, None)
        if x0.sum() > m:
            x0 = x0 * (m / x0.sum())
        res = minimize(neg_val, x0, method="SLSQP", bounds=slsqp_bounds,
                        constraints=cons, options=dict(maxiter=500, ftol=1e-14))
        cand_val = -res.fun if res.success else val
        if cand_val > best_val:
            best_val, best_x = cand_val, res.x if res.success else x0
        if verbose:
            print(f"      candidate start val={val:+.4e} -> polished {cand_val:+.4e}")

    return best_val, best_x


# ===========================================================================
# Grid CSV I/O and sweep
# ===========================================================================

def load_mstar_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(
                s=int(row["s"]), l=int(row["l"]), n=int(row["n"]),
                mstar=int(row["mstar"]), delta_max=float(row["delta_max"]),
            ))
    return rows


OUT_FIELDS = ["s", "l", "n", "mstar", "m", "delta", "max_value", "verdict"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--interior", type=int, default=200_000)
    parser.add_argument("--face", type=int, default=50_000)
    parser.add_argument("--de-seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    grid_rows = load_mstar_rows(args.csv)
    rng = np.random.default_rng(args.seed)

    file_exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
    done = set()
    if file_exists:
        with open(args.out, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["s"]), int(row["l"]), int(row["n"]), int(row["m"])))

    print(f"Loaded {len(grid_rows)} (s,l,n) rows from {args.csv}; "
          f"{len(done)} (s,l,n,m) instances already done in {args.out}\n")

    header = f"{'(s,l,n,m)':<14}{'m*':>4}{'delta':>10}{'max value':>16}{'verdict':>12}"
    print(header)
    print("-" * len(header))

    with open(args.out, "a" if file_exists else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
        if not file_exists:
            wr.writeheader()
            fh.flush()

        for row in grid_rows:
            s, l, n, mstar, delta = row["s"], row["l"], row["n"], row["mstar"], row["delta_max"]
            for m in range(2, mstar + 1):
                key = (s, l, n, m)
                if key in done:
                    continue

                game = Bertrand(s=s, l=l, n=n, m=m)
                x_star = game.bne

                grid_n = {1: 20_000, 2: 3000, 3: 220, 4: 40, 5: 19}.get(m)

                best_val, best_x = robust_max_minty(
                    game, x_star, m, delta, rng,
                    n_interior=args.interior, n_face=args.face,
                    n_de_seeds=args.de_seeds, grid_n=grid_n, verbose=args.verbose)

                verdict = "VIOLATES" if best_val > args.tol else "satisfies"
                print(f"({s},{l},{n},{m})".ljust(14) + f"{mstar:>4}{delta:>10.4f}"
                      f"{best_val:>16.6e}{verdict:>12}")

                wr.writerow(dict(s=s, l=l, n=n, mstar=mstar, m=m, delta=delta,
                                  max_value=best_val, verdict=verdict))
                fh.flush()

    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()

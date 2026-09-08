"""Regression checks for the retained model and rescaled SOS primitives."""
from pathlib import Path
import csv
import sys

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/general_lyapunov"))
sys.path.insert(0, str(ROOT / "scripts/geometry"))

from sos_common import (
    build_game, build_metric_matrix, build_stacked_lyapunov, sdp_size,
)
from mstar_delta_grid import compute_mstar
from solver_rescaled import (
    boundary_intrinsic_degree_rescaled, build_sdp_problem_rescaled,
    decrease_intrinsic_degree_rescaled, multiplier_degree_floor, spectral_rate,
)


@pytest.mark.parametrize("s", [0, 1, 2])
def test_equilibrium_and_candidate_vanish(s):
    x, v, xstar, _ = build_game(s, 1, 2, 2)
    at_star = dict.fromkeys(x, xstar)
    assert all(sp.simplify(vi.subs(at_star)) == 0 for vi in v)
    L, gradient, *_ = build_stacked_lyapunov(x, 2, 4, xstar)
    assert sp.expand(L.subs(at_star)) == 0
    assert all(sp.expand(g.subs(at_star)) == 0 for g in gradient)


@pytest.mark.parametrize("intrinsic,expected", [
    (-1, 0), (0, 0), (1, 0), (2, 2), (5, 4), (6, 6),
])
def test_multiplier_degree_floor(intrinsic, expected):
    """The floor is the smallest even degree at least intrinsic-1."""
    floor = multiplier_degree_floor(intrinsic)
    assert floor == expected
    assert floor % 2 == 0 and floor >= 0


def test_spectral_rate_is_hurwitz_and_rational():
    x, v, xstar, _ = build_game(2, 1, 2, 2)
    Hinv = build_metric_matrix(2, 1, 2, 2, "q", q=2)
    r, alpha = spectral_rate(x, v, xstar, Hinv)
    assert alpha < 0
    assert r.is_Rational and r > 0
    assert abs(float(r) - abs(alpha)) <= 1e-6


def test_rescaled_problem_builds_at_archived_size():
    """s=2, l=1, n=2, m=2, q=2 is the smallest archived row: 199 variables."""
    d_L, m = 2, 2
    d_dec = multiplier_degree_floor(
        decrease_intrinsic_degree_rescaled(2, 1, 2, m, d_L, "q", q=2))
    d_bnd = multiplier_degree_floor(
        boundary_intrinsic_degree_rescaled(2, 1, 2, m, d_L, "q", q=2))
    assert (d_dec, d_bnd) == (4, 0)
    built = build_sdp_problem_rescaled(
        2, 1, 2, m, d_L, d_dec, d_bnd, 1e-4, 1.0, sp.Rational(1, 250),
        metric="q", q=2)
    prob, L, x, xstar = built[:4]
    assert sp.expand(L.subs(dict.fromkeys(x, xstar))) == 0
    assert prob.all_sos_constraints
    assert sdp_size(prob) == 199


def test_only_retained_metrics_are_accepted():
    with pytest.raises(ValueError):
        build_metric_matrix(0, 1, 2, 2, "unknown", q=2)


@pytest.mark.parametrize("bad", [{"c": 0.0}, {"c": 2.0}, {"tau": 0.0}])
def test_rescaled_solver_rejects_out_of_range_constants(bad):
    from solver_rescaled import solve_sdp_rescaled
    kwargs = dict(s=2, l=1, n=2, m=2, q=2, d_L=2, d_dec=4, d_bnd=0)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        solve_sdp_rescaled(metric="q", verbose=False, **kwargs)


def test_all_archived_minty_thresholds():
    with (ROOT / "results/mstar_delta_grid.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 36
    for row in rows:
        s, l, n = (int(row[key]) for key in ("s", "l", "n"))
        assert compute_mstar(l * (n - 1) + s, l * n) == int(row["mstar"])

"""Regression checks for the retained model and exact certificate primitives."""
from pathlib import Path
import csv
import sys

import pytest
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/general_lyapunov"))
sys.path.insert(0, str(ROOT / "scripts/geometry"))

from sos_common import build_game, build_metric_matrix, build_stacked_lyapunov
from mstar_delta_grid import compute_mstar
from solver_deflated import (
    _project_to_affine_exact, decrease_rate_term, exact_rationalize_L,
    is_psd_exact, build_sdp_problem_v2,
)


@pytest.mark.parametrize("s", [0, 1, 2])
def test_equilibrium_and_candidate_vanish(s):
    x, v, xstar, _ = build_game(s, 1, 2, 2)
    at_star = dict.fromkeys(x, xstar)
    assert all(sp.simplify(vi.subs(at_star)) == 0 for vi in v)
    L, gradient, *_ = build_stacked_lyapunov(x, 2, 4, xstar)
    assert sp.expand(L.subs(at_star)) == 0
    assert all(sp.expand(g.subs(at_star)) == 0 for g in gradient)


@pytest.mark.parametrize("matrix,expected", [
    ([[2, 1], [1, 2]], True), ([[1, 1], [1, 1]], True),
    ([[1, 2], [2, 1]], False), ([[1, 1], [0, 1]], False),
])
def test_exact_psd(matrix, expected):
    assert is_psd_exact(sp.Matrix(matrix))[0] is expected


def test_rational_projection_preserves_identity():
    B = sp.Matrix([[1, 2, 0], [0, 1, 1]])
    b = sp.Matrix([sp.Rational(1, 3), sp.Rational(2, 7)])
    projected = _project_to_affine_exact(sp.zeros(3, 1), B, b)
    assert B * projected == b
    assert all(value.is_Rational for value in projected)


def test_archived_rate_forms_are_distinct():
    x = sp.symbols("x1:3", positive=True)
    p = sp.prod(x)
    L = sum((xi - sp.Rational(1, 2)) ** 2 for xi in x)
    assert sp.expand(decrease_rate_term(x, 2, L, p, "L") - p * L) == 0
    assert sp.expand(decrease_rate_term(x, 2, L, p, "pL") - p**2 * L) == 0
    with pytest.raises(ValueError):
        decrease_rate_term(x, 2, L, p, "invalid")


@pytest.mark.parametrize("rate", ["L", "pL"])
def test_deflated_problem_builds_and_remembers_grams(rate):
    built = build_sdp_problem_v2(2, 1, 2, 2, 2, 6, 2, 2, decrease_rate=rate)
    prob, L, x, xstar = built[:4]
    assert sp.expand(L.subs(dict.fromkeys(x, xstar))) == 0
    assert prob.all_sos_constraints
    assert all(hasattr(con, "param_poly") for con in prob.all_sos_constraints)


def test_only_retained_metrics_are_accepted():
    with pytest.raises(ValueError):
        build_metric_matrix(0, 1, 2, 2, "unknown", q=2)


def test_all_archived_minty_thresholds():
    with (ROOT / "results/mstar_delta_grid.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 36
    for row in rows:
        s, l, n = (int(row[key]) for key in ("s", "l", "n"))
        assert compute_mstar(l * (n - 1) + s, l * n) == int(row["mstar"])


def test_sweep_imports_and_rate_choices():
    from sweep_deflated import build_parser
    for rate in ("L", "pL"):
        args = build_parser().parse_args(["--decrease-rate", rate])
        assert args.decrease_rate == rate
        assert args.m_values == (2,)

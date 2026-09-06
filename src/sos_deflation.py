"""Rescaled polynomial bases, degree reduction, and SOS margin constraints."""
import numpy as np
import picos as pic
import sympy as sp
from SumOfSquares.basis import Basis

def domain_scale(xstar, lower_bound, rho=None):
    """Exact-rational scale rho for z = (x-x*)/rho, so every basis here is
    O(1) around 0 rather than around x*.

    Defaults to xstar - lower_bound.  This is a conditioning device, not a
    rigorous fit of the domain (a box cut by a simplex constraint, not
    symmetric about x*): nothing downstream needs z inside [-1,1], only
    that it is not absurdly mis-scaled.
    """
    if rho is None:
        rho = xstar - sp.Rational(str(lower_bound))
    else:
        rho = sp.Rational(str(rho))
    if rho <= 0:
        raise ValueError(f"domain_scale: rho={rho} must be positive.")
    return rho


def build_z(x, xstar, rho):
    """Fresh z_1..z_m symbols and the substitution dict x_i -> rho*z_i+x*
    (used to move an already-built x-based expression into z)."""
    m = len(x)
    z = sp.symbols(f"z1:{m + 1}", real=True)
    x_in_terms_of_z = {xi: rho * zi + xstar for xi, zi in zip(x, z)}
    return z, x_in_terms_of_z


def basis_no_low_degree(nvars, k, min_deg):
    """Basis of ALL monomials in nvars variables of degree min_deg..k
    (degrees 0..min_deg-1 excluded entirely) -- the same construction
    Basis.from_degree(nvars, k) uses, filtered."""
    full = Basis.from_degree(nvars, k)
    return Basis([mono for mono in full.monoms if sum(mono) >= min_deg])


def poly_variable_no_low_degree(name, variables, deg, min_deg):
    """Like SumOfSquares.basis.poly_variable, but every monomial of degree
    < min_deg is OMITTED -- no free coefficient exists for it, so the
    polynomial vanishes to that order identically rather than only at the
    (rounding-fragile) optimum.  Used with min_deg=2 for Decrease's chi_i;
    each chi_i then vanishes to order at least two at x*.
    """
    variables = sorted(variables, key=str)
    basis = basis_no_low_degree(len(variables), deg, min_deg)
    coeffs = sp.symbols(f"{name}_:{len(basis)}")
    return sum(coeff * sp.prod([v ** p for v, p in zip(variables, monom)])
               for monom, coeff in zip(basis, coeffs))


def _homogeneous_part(expr, z_list, deg):
    """dict {monomial: coefficient} for the EXACTLY-degree-`deg`
    homogeneous part of expr (a polynomial in z_list whose coefficients
    may themselves be free sympy symbols, e.g. straight out of
    poly_variable/poly_variable_no_low_degree -- nothing here requires
    them to be numeric)."""
    poly = sp.Poly(sp.expand(expr), *z_list)
    return {mono: coeff for mono, coeff in zip(poly.monoms(), poly.coeffs())
           if sum(mono) == deg}


def eliminate_forced_top_degree(mult_list, g_list, z_list, mult_deg):
    """Eliminate coefficients forced to vanish by the parity of an SOS polynomial.

    Every g_j is affine, so mult_j*g_j reaches degree mult_deg+1.  Once
    mult_deg+1 exceeds expr_z's own degree (the normal case as soon as the
    multiplier degree is raised past the feasibility floor for margin),
    that becomes target_z's actual top degree with nothing above it.  An
    SOS polynomial cannot have a nonzero ODD top-degree term, so when
    mult_deg+1 is odd that part is FORCED to vanish identically -- for
    every feasible point, not just this instance.

    Left implicit it is only ever satisfied to solver tolerance: measured
    directly, every degree-(mult_deg+1) coefficient sat at 1e-7..1e-9
    while the genuinely free ones sat at order 1.  That residual is
    exactly the near-zero margin a Stage-2 re-solve reports.  It is a
    forced-zero COEFFICIENT, distinct from the forced-zero POINT that (a)
    fixes, and it affects Decrease and Boundary alike.

    Fix: the vanishing condition is linear in the multipliers' leading
    coefficients and involves only each g_j's known linear part -- never M
    -- so solve it once by exact rational rref and substitute back before
    any SOS constraint is built.  target_z's true degree then becomes even
    by construction.

    This does not shrink the feasible set: the identity already held at
    every feasible point.  It reparametrises the same variety with fewer,
    exactly-consistent coordinates.

    Returns {symbol: linear_expr}, or {} if mult_deg+1 is even (nothing is
    forced) or every g_j is constant (defensive; never happens here).
    """
    top_deg = mult_deg + 1
    if top_deg % 2 == 0:
        return {}

    per_mult_top = [_homogeneous_part(mj, z_list, mult_deg) for mj in mult_list]
    top_coeffs = [c for top in per_mult_top for c in top.values()]
    if not top_coeffs:
        return {}

    total = sp.Integer(0)
    for mj_top, gj in zip(per_mult_top, g_list):
        g_lin = _homogeneous_part(gj, z_list, 1)
        if not g_lin:
            continue
        top_expr = sum(c * sp.prod([zi ** e for zi, e in zip(z_list, mono)])
                       for mono, c in mj_top.items())
        g_lin_expr = sum(c * sp.prod([zi ** e for zi, e in zip(z_list, mono)])
                         for mono, c in g_lin.items())
        total += sp.expand(top_expr * g_lin_expr)

    if total == 0:
        return {}

    top1 = sp.Poly(total, *z_list)
    equations = [coeff for mono, coeff in zip(top1.monoms(), top1.coeffs())
                if sum(mono) == top_deg]
    if not equations:
        return {}

    A, _ = sp.linear_eq_to_matrix(equations, top_coeffs)
    rref, pivots = A.rref()
    subs = {}
    for row_idx, col_idx in enumerate(pivots):
        expr = sp.Integer(0)
        for k in range(len(top_coeffs)):
            if k == col_idx:
                continue
            coeff = rref[row_idx, k]
            if coeff != 0:
                expr -= coeff * top_coeffs[k]
        subs[top_coeffs[col_idx]] = sp.expand(expr)
    return subs


def _solve_max_margin(prob, mu_pic, solver, timelimit=None, solve_options=None,
                      fallback_solver="cvxopt"):
    """Solve by MAXIMISING mu rather than a bare "find".

    A "find" objective was observed to land on the identical marginal point
    regardless of how much margin was technically available -- an
    interior-point solver has no reason to prefer the interior unless the
    objective rewards it.  Maximising mu gives it one.

    solve_options is forwarded to PICOS' Problem.solve.  For MOSEK this is
    where callers can pass raw mosek_params.  Use raw MOSEK interior-point
    parameters for very tight tolerances: PICOS' generic abs_*_fsb_tol also
    touches simplex basis tolerances, whose legal lower bounds can be looser.

    Falls back only when fallback_solver is not None and the requested solver
    errors out. A
    returned "unknown" is not evidence of infeasibility.
    """
    import picos
    prob.set_objective("max", mu_pic)
    kwargs = {} if timelimit is None else {"timelimit": timelimit}
    if solve_options:
        kwargs.update(solve_options)
    try:
        sol = prob.solve(solver=solver, **kwargs)
        return True, solver, getattr(sol, "primalStatus", "unknown")
    except picos.modeling.problem.SolutionFailure as exc:
        print(f"  [{solver}] {exc}")
        return False, solver, str(exc)
    except Exception as exc:
        if fallback_solver is None:
            print(f"  [{solver}] failed ({exc!r}); no fallback requested.")
            return False, solver, str(exc)
        print(f"  [{solver}] failed ({exc!r}); falling back to {fallback_solver}.")
        try:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("mosek_params", None)
            sol = prob.solve(solver=fallback_solver, **fallback_kwargs)
            return True, fallback_solver, getattr(sol, "primalStatus", "unknown")
        except picos.modeling.problem.SolutionFailure as exc2:
            print(f"  [{fallback_solver}] {exc2}")
            return False, fallback_solver, str(exc2)


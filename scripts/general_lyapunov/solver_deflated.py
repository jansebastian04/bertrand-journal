"""Deflated SOS certificates and exact rational verification for q=0,1,2.

This modifies the floating-point frontier SDP in solver_rescaled.py to make
exact verification possible. Currently only a small subset of instances
can be verified; these verified results do not cover the full frontier.

Use z=(x-x*)/rho and the full monomial basis of degrees 1..d_L/2 with
relative scaling. L=Phi(z)^T M Phi(z) vanishes with its gradient at x*.
The SDP fixes mu and eps_M, imposes mu*I<=M<=kappa*I, and maximizes
eps_R, an absolute residual Gram floor Q>=eps_R*I. Multiplier Gram
matrices have fixed absolute floors eps_M and upper bounds mult_cap.

Decrease rate L certifies dL/dt<=-c*r*L; pL certifies dL/dt<=-c*r*p*L,
where p=prod(x), r=abs(alpha(Hinv(x*) Dv(x*))), Hinv=diag(x_i**(2-q)).
None selects pL for q<2 and L for q=2. Boundary constraints use face
substitution by default; an ambient polynomial formulation is optional.

Exact verification reconstructs rational Gram matrices, checks polynomial
identities, and proves positive semidefiniteness. A floating eigenvalue
screen can reject but cannot certify a matrix. Skipped checks do not verify.
The lower candidate LMI gates verification; the upper cap is normalization.

Shared construction helpers live in src/sos_common.py and sos_deflation.py.
Exact-verification primitives are defined directly here. The sweep CLI
records the chosen rate, denominator bounds, degrees, and verified L.
"""

import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import picos as pic
import sympy as sp

from SumOfSquares.basis import Basis, poly_variable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sos_common import DEMAND_NAMES, Q_NAMES, BertrandSOSProblem, _total_degree, build_game, build_metric_matrix, define_constraints, sdp_size, rationalize_scalar
from solver_rescaled import (
    RESULTS_DIR,
    spectral_rate,
    multiplier_degree_floor,
)
from sympy.matrices.exceptions import NonPositiveDefiniteMatrixError


def is_psd_exact(M):
    """Test symmetry and exact positive semidefiniteness over rational entries.
    
    Try SymPy LDL decomposition first, then its semidefinite predicate for
    singular matrices. Return (bool, reason)."""
    A = sp.Matrix(M)
    if A != A.T:
        return False, "not symmetric"

    try:
        A.LDLdecomposition()
        return True, "PSD (positive definite, via sympy LDLdecomposition)"
    except NonPositiveDefiniteMatrixError:
        pass

    if A.is_positive_semidefinite:
        return True, "PSD (rank deficient, via sympy is_positive_semidefinite)"
    return False, "not PSD (via sympy is_positive_semidefinite)"


def _bareiss_linsolve(A, c):
    """Solve A*w=c exactly for a full-rank integer A and rational c.
    
    DomainMatrix.solve_den returns a numerator and shared denominator.
    Raise ValueError for a singular system."""
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError

    n = A.rows
    dm_A = DomainMatrix.from_Matrix(A).convert_to(sp.ZZ)
    dm_c = DomainMatrix.from_Matrix(sp.Matrix(c)).convert_to(sp.QQ)
    try:
        xnum, xden = dm_A.solve_den(dm_c)
    except DMNonInvertibleMatrixError as exc:
        raise ValueError(f"A is singular (or needs column pivoting): {exc}")
    return xnum.to_Matrix() / sp.sympify(xden)


def _project_to_affine_exact(x0, B, b):
    """Nearest exact rational point (Euclidean norm) to x0 among all exact
    vectors satisfying B x = b exactly.

    Closed form Q_final = x0 + B^T (B B^T)^{-1} (b - B x0), valid whenever B
    has full row rank (checked implicitly: _bareiss_linsolve raises
    otherwise, which would indicate a redundant/inconsistent constraint set
    rather than a property of the SDP solution). x0, B, b must already be
    exact rationals. B B^T is solved via _bareiss_linsolve rather than
    Matrix.LUsolve -- see that function's docstring for why the naive
    solve's denominator blowup was making exact verification hang.
    """
    residual = b - B * x0
    BBt = B * B.T
    w = _bareiss_linsolve(BBt, residual)
    return x0 + B.T * w


def _gram_unknown_index(n):
    """Map (i,j), i<=j, to a linear index into the upper-triangular vector
    of an n x n symmetric matrix, in row-major order of (i,j)."""
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i, n):
            idx[(i, j)] = k
            k += 1
    return idx


def _gram_equation_matrix(basis):
    """Structural (basis-only, polynomial-independent) constraint matrix A
    with A[row, idx[i,j]] = multiplicity of the unordered pair {i,j} among
    basis.sos_sym_entries[monomial], for each monomial (row) representable
    by the basis. A is always full row rank: for any right-hand side b,
    distributing b[row] evenly across the pairs of that row's monomial is
    an explicit preimage.
    """
    n = len(basis)
    unk = _gram_unknown_index(n)
    monoms = sorted(basis.sos_sym_entries.keys())
    A = sp.zeros(len(monoms), len(unk))
    for row, mono in enumerate(monoms):
        for (i, j) in basis.sos_sym_entries[mono]:
            A[row, unk[(min(i, j), max(i, j))]] += 1
    return A, monoms, unk


def _vec_to_sym_matrix(vec, n, unk):
    M = sp.zeros(n, n)
    for (i, j), k in unk.items():
        M[i, j] = M[j, i] = vec[k]
    return M


def exact_gram_matrix(con, rational_vals, q_max_den=10 ** 10):
    """Exact rational Gram matrix for one SOS constraint, given rounded
    rational values for every free SDP variable (including M's own
    entries, already rationalised -- see verify_certificate_exact_
    exact_rationalize_L).

    Returns (Q_final, identity_ok), where identity_ok re-checks
    A @ vec(Q_final) == b directly (cheap: this equality is exactly the
    statement that b(x)^T Q_final b(x) reproduces the rationalised target
    polynomial coefficient-by-coefficient), as a defence against a coding
    error in this function rather than as part of the proof itself.

    Q, unlike L/the multiplier coefficients, is not a reported quantity --
    it only has to be *some* exact matrix on the correct side of the cone
    -- so q_max_den is set far tighter than the max_den used for the
    decision variables: rounding Q0 to a short rational perturbs its
    entries by up to ~1/(2*q_max_den), and if that exceeds the solver's
    own accuracy (~1e-10) it can turn a merely-noisy Q0 into one with a
    spuriously larger negative eigenvalue after the affine correction
    below. 1e10 is tight enough to preserve the solver's noise floor while
    keeping the arithmetic tractable.
    """
    n = len(con.basis)
    A, monoms, unk = _gram_equation_matrix(con.basis)
    ordered = sorted(unk.items(), key=lambda kv: kv[1])

    target = {mono: coeff.subs(rational_vals)
              for mono, coeff in zip(con.param_poly.monoms(), con.param_poly.coeffs())}
    b = sp.Matrix([target.get(mono, sp.Integer(0)) for mono in monoms])

    # A 1x1 SymmetricVariable's .value comes back 0-dimensional from PICOS;
    # reshape so Q0[i, j] indexing below is always valid.
    Q0 = np.asarray(con.Qval, dtype=float).reshape(n, n)
    q0 = sp.Matrix([rationalize_scalar((Q0[i, j] + Q0[j, i]) / 2.0, q_max_den)
                    for (i, j), _ in ordered])

    q_final = _project_to_affine_exact(q0, A, b)
    identity_ok = all(c == 0 for c in (A * q_final - b))
    Q_final = _vec_to_sym_matrix(q_final, n, unk)
    return Q_final, identity_ok


def exact_rationalize_L(L, x, xstar, rational_vals, max_den=10 ** 6):
    """Exactly rationalise L's coefficients (M's entries) -- no affine
    re-projection needed.

    L is built by build_stacked_lyapunov from Phi(y)^T M Phi(y) with every
    entry of Phi vanishing at y=0, so L(x*) = 0 and grad L(x*) = 0 hold
    identically for every choice of M (see that function's docstring).
    There is no affine subspace to project onto here: each rounded
    rational coefficient can be substituted directly and both identities
    still hold exactly. Returns (L_exact, updated_rational_vals,
    identity_ok), where identity_ok is a direct symbolic re-check of both
    identities with the rationalised coefficients -- a defence against a
    coding error in build_stacked_lyapunov, not part of the proof itself
    (the vanishing is unconditional on the coefficients).
    """
    x_list = list(x)
    L_syms = sorted(L.free_symbols - set(x_list), key=str)

    updated = dict(rational_vals)
    updated.update({sym: rationalize_scalar(rational_vals[sym], max_den)
                    for sym in L_syms})
    L_exact = sp.expand(L.subs(updated))

    xstar_subs = {xi: xstar for xi in x_list}
    grad_L_exact = [sp.diff(L_exact, xi) for xi in x_list]
    identity_ok = (L_exact.subs(xstar_subs) == 0
                   and all(gi.subs(xstar_subs) == 0 for gi in grad_L_exact))
    return L_exact, updated, identity_ok

# Shared polynomial-basis and margin primitives.
from sos_deflation import domain_scale, build_z, basis_no_low_degree, poly_variable_no_low_degree, eliminate_forced_top_degree, _solve_max_margin


# ===========================================================================
# Default scale, rationalization bounds, and fixed margins.
# The sweep overrides denominator bounds and fixed margins per attempt.
# ===========================================================================

DELTA = sp.Rational(1, 250)
KAPPA = 1.0
TAU = 1e-3
MAX_DEN = 10 ** 6
EPS_M = 100.0 / MAX_DEN          # = 1e-4, comfortably above rounding noise
RATE_FORMS = ("L", "pL")
MU_FIXED = EPS_M                 # mu is no longer maximised, see below
C_RATE = 1.0                     # the decrease-rate fraction c in (0,2).
#   Measured the best available value on every axis: it is the only one at
#   which all three regularisers solve at all (q=1 goes 'unknown' below
#   c ~ 1e-3), it gives q=1 its best surviving-block count, and the rate it
#   certifies, c*r*delta^m, is six orders larger than at c=1e-7.  Smaller c
#   buys SOLVABILITY headroom in some formulations but never roundability:
#   at q=2 every c from 1 to 1e-9 verifies 13/13 alike.
Q_MAX_DEN = 10 ** 10

# The rate weight is decided by rate_weight_needed(q); None means "from q".
RATE_WEIGHT_AUTO = None


# ===========================================================================
# Lyapunov candidate: L = Phi^T M Phi with Phi = W * (deflated monomials in z)
# ===========================================================================

def domain_box(m, lower_bound):
    """(lo, hi): the exact rational interval every x_i is confined to by
    g >= 0, namely [delta, m - (m-1)*delta] -- each coordinate is at least
    delta, and the simplex constraint sum_k x_k <= m then caps it once the
    other m-1 coordinates sit at their own floor."""
    lo = sp.Rational(lower_bound)
    return lo, sp.Integer(m) - (m - 1) * lo


def z_amplitude(m, xstar, rho, lower_bound):
    """Zmax = max over the domain box of |z_i| = |x_i - x*|/rho.  Exact
    rational.  The lower end contributes exactly 1 (rho = x* - delta), so
    this is >= 1 always."""
    lo, hi = domain_box(m, lower_bound)
    return max(abs(lo - xstar) / rho, abs(hi - xstar) / rho)


def metric_amplitude(q, m, lower_bound):
    """Hmax = max over the domain box of max_i Hinv[i,i] = max_i x_i^{2-q}.

    x^{2-q} is INCREASING in x for q < 2 and DECREASING for q > 2, so the
    maximiser is the top of the box in the first case and the bottom in the
    second.  Getting this backwards would only mis-scale the "absolute"
    rescaling, but it would do so by a factor of ((m-(m-1)delta)/delta)^{q-2}
    -- 250 at q=3, delta=1/250 -- so it is worth being exact about.
    """
    lo, hi = domain_box(m, lower_bound)
    return hi ** (2 - q) if q <= 2 else lo ** (2 - q)


def clearing_power(q):
    """The exponent j such that p(x)^j * Hinv(x) * v(x) is POLYNOMIAL, where
    p(x) = prod_k x_k.

    Each component of the game gradient has the shape v_i = N_i/(c*x_i) with
    N_i polynomial, so Hinv[i,i]*v_i = x_i^{1-q}*N_i/c and multiplying by
    p^j contributes x_i^{j}: polynomial iff j >= q-1.  j = 1 (plain
    p) therefore covers q in {0,1,2} and NOT q=3, where Hinv = diag(1/x_i)
    leaves an x_i^{-2} that a single p cannot clear.

    Returns max(1, q-1): at least one p is always needed for v's own
    denominator, and q-1 of them once the metric contributes negative
    powers of its own.
    """
    return max(1, q - 1)


def scaled_dynamics_poly_q(x, v, Hinv, q):
    """Clear the q-metric dynamics denominators using a power of prod(x)."""
    m = len(v)
    P = sp.prod(x) ** clearing_power(q)
    Pv = [sp.expand(v[j] * P) for j in range(m)]
    wtilde = [sp.expand(sum(Hinv[i, j] * Pv[j] for j in range(m))) for i in range(m)]
    for i, e in enumerate(wtilde):
        if not e.is_polynomial(*x):
            raise ValueError(
                f"scaled_dynamics_poly_q: wtilde[{i}] is not polynomial at q={q} "
                f"even after multiplying by p^{clearing_power(q)}.")
    return wtilde, P


def lyapunov_monomials(m, d_L, lyap_basis):
    """The exponent tuples of Phi, all of total degree >= 1.

    "full"    every monomial of degree 1..K -- the same deflated basis
              basis_no_low_degree builds for the Decrease Gram matrices,
              so M and Q share one basis family.
    "powers"  stacked pure powers z_i^k only.
    """
    if d_L % 2 != 0:
        raise ValueError(f"d_L must be even; got d_L={d_L}")
    K = d_L // 2
    if lyap_basis == "full":
        return list(basis_no_low_degree(m, K, min_deg=1).monoms)
    if lyap_basis == "powers":
        return [tuple(k if i == j else 0 for j in range(m))
                for k in range(1, K + 1) for i in range(m)]
    raise ValueError(f"unknown lyap_basis {lyap_basis!r}; expected 'full' or 'powers'.")


def basis_weights(monoms, zmax, hmax, basis_rescale, max_den=MAX_DEN):
    """W's diagonal, one exact rational per basis monomial.

    basis_rescale = False        -> all ones, basis.
    basis_rescale = "absolute"   -> w_a = 1 / ( zmax^{|a|} * sqrt(hmax) ),
                                   the literal form of the rescaling remark.
    basis_rescale = "relative"   -> w_a = zmax^{1-|a|}, the same weights
                                   NORMALISED so the degree-1 monomials keep
                                   weight 1.

    Prefer "relative".  sqrt(hmax) is a UNIFORM factor across the basis, so
    it contributes exactly nothing to M's conditioning -- it cancels out of
    every constraint that is homogeneous in L (Decrease in every "*L" rate
    form, and Boundary's inner product up to the flat mu).  What it does do
    is shrink L against the FIXED cap M <= kappa*I, and therefore shrink the
    reported margin: measured at d_L=2, q=2, where W is a scalar and the
    rescaling can have no conditioning effect at all, "absolute" still drops
    mu from 0.349 to 0.192 purely through that units change.  The genuine
    conditioning content of the remark is the zmax^{|a|} SPREAD across
    degrees, which "relative" keeps and "absolute" merely offsets.

    Rationalising each weight is what keeps L exactly rational-linear in M,
    so the exact verification pass needs no un-scaling step of its own:
    L = Phi^T (W M W) Phi has rational coefficients in M's entries by
    construction.
    """
    if not basis_rescale:
        return [sp.Integer(1)] * len(monoms)
    mode = "absolute" if basis_rescale is True else basis_rescale
    if mode not in ("absolute", "relative"):
        raise ValueError(f"basis_rescale must be False/'absolute'/'relative'; "
                         f"got {basis_rescale!r}.")
    zmax = sp.Rational(zmax)
    if mode == "relative":
        return [rationalize_scalar(float(zmax ** (1 - sum(a))), max_den)
                for a in monoms]
    inv_sqrt_h = rationalize_scalar(1.0 / float(sp.sqrt(hmax)), max_den)
    return [rationalize_scalar(float(inv_sqrt_h / zmax ** sum(a)), max_den)
            for a in monoms]


def build_lyapunov(x, m, d_L, xstar, rho, lyap_basis="full", weights=None):
    """Build L=Phi(z)^T M Phi(z) and its x-gradient.

    The basis has no constant term, so L and its gradient vanish at x*.
    Return (L, grad_L, M_sym, monoms, weights, dim)."""
    monoms = lyapunov_monomials(m, d_L, lyap_basis)
    if weights is None:
        weights = [sp.Integer(1)] * len(monoms)
    z = [(xi - xstar) / rho for xi in x]
    Phi = [w * sp.prod([zi ** e for zi, e in zip(z, a)])
           for w, a in zip(weights, monoms)]
    dim = len(Phi)

    entries = sp.symbols(f"M_0:{dim * (dim + 1) // 2}")
    it = iter(entries)
    M_sym = sp.zeros(dim, dim)
    for i in range(dim):
        for j in range(i, dim):
            sij = next(it)
            M_sym[i, j] = sij
            M_sym[j, i] = sij

    Phi_col = sp.Matrix(Phi)
    L = sp.expand((Phi_col.T * M_sym * Phi_col)[0, 0])
    grad_L = [sp.diff(L, xi) for xi in list(x)]
    return L, grad_L, M_sym, monoms, weights, dim


# ===========================================================================
# Row scaling of the Gram coefficient-matching system
#
# The matching equations  sum_{(i,j) in pairs(alpha)} Q[i,j] == coeff_alpha
# are handed to mosek exactly as written, with whatever magnitude each
# monomial's coefficient happens to have.  Those magnitudes differ by orders
# of magnitude across alpha, and mosek equalises its residual in the SCALED
# units it is given -- so the rows with large coefficients absorb the
# accuracy budget and the rest are met only loosely.  Since the exact
# verification has to CLOSE whatever residual is left (it projects onto the
# exact slice, moving lambda_min by roughly that much), the largest residual
# over all rows is precisely what decides whether rounding survives.
#
# Multiplying a matching equation through by any w > 0 leaves the feasible
# set identical and changes only what mosek considers "close enough" on it.
# ===========================================================================

def matching_row_scales(poly, basis, mode="l1", floor=1e-12):
    """One positive weight per matching row (per representable monomial).

    mode="l1" (default): coeff_alpha is AFFINE in the free SDP variables,
        coeff_alpha = c_0 + sum_k c_k * sym_k.  The row's natural scale is
        the l1 norm |c_0| + sum_k |c_k| of that affine form, and dividing by
        it puts every row on the same footing -- textbook row equilibration
        for an affine system, applied to the one system whose residual we
        have measured to be decisive.
    mode="rownorm": 1/sqrt(#pairs), equilibrating the CONSTRAINT MATRIX's
        rows instead of the right-hand side.  Cheaper, ignores the data.
    mode=None: all ones (unscaled basis).

    Returns {monomial: weight}.
    """
    if not mode:
        return {}
    coeffs = dict(zip(poly.monoms(), poly.coeffs()))
    scales = {}
    for mono, pairs in basis.sos_sym_entries.items():
        if mode == "rownorm":
            scales[mono] = 1.0 / max(len(pairs), 1) ** 0.5
            continue
        c = coeffs.get(mono, sp.Integer(0))
        expr = sp.expand(c)
        syms = sorted(expr.free_symbols, key=str)
        if syms:
            poly_c = sp.Poly(expr, *syms)
            l1 = sum(abs(float(t)) for t in poly_c.coeffs())
        else:
            l1 = abs(float(expr))
        scales[mono] = 1.0 / max(l1, floor)
    return scales


def add_sos_constraint_scaled(prob, expr, variables, name="", vanishing=False,
                              row_scaling=None):
    """add_sos_constraint with the matching rows individually rescaled.

    vanishing=True reproduces add_sos_constraint_vanishing (the Gram
    basis excludes the constant monomial, for a polynomial proven to vanish
    to order 2 at x*); vanishing=False reproduces the plain full-basis form.
    Both are otherwise unchanged -- only the equations' scaling differs, and
    that changes no feasible point, only mosek's notion of how tightly to
    meet each one.
    """
    from SumOfSquares.SoS import SOSConstraint

    prob._sos_const_count += 1
    name = name or f"_Q{prob._sos_const_count}"
    variables_sorted = sorted(variables, key=str)
    poly = sp.poly(expr, variables_sorted)
    deg = poly.total_degree()
    half = (deg + 1) // 2
    basis = (basis_no_low_degree(len(variables_sorted), half, min_deg=1)
             if vanishing else Basis.from_degree(len(variables_sorted), half))

    if vanishing:
        for mono in poly.monoms():
            if mono not in basis.sos_sym_entries:
                raise ValueError(
                    f"add_sos_constraint_scaled: monomial {mono} in {name!r} is not "
                    f"representable by the constant-excluded basis -- expr does not "
                    f"actually vanish to the expected order.")

    mono_to_coeffs = dict(zip(poly.monoms(), map(prob.sp_to_picos, poly.coeffs())))
    scales = matching_row_scales(poly, basis, mode=row_scaling)

    Q = pic.SymmetricVariable(name, len(basis))
    for mono, pairs in basis.sos_sym_entries.items():
        coeff = mono_to_coeffs.get(mono, 0)
        w = scales.get(mono, 1.0)
        lhs = sum(Q[i, j] for i, j in pairs)
        prob.add_constraint(w * lhs == w * coeff)

    pic_const = prob.add_constraint(Q >> 0)
    con = SOSConstraint(pic_const, Q, basis, variables_sorted, deg)
    con.param_poly = poly
    prob.all_sos_constraints.append(con)
    return con


def putinar_bounded_scaled(prob, expr_x, g_x, z_list, x_in_terms_of_z, d, tag,
                           mult_cap, free_face=None, vanishing_at_xstar=False,
                           row_scaling=None):
    """Build a Putinar certificate in z with bounded multiplier Gram matrices.

    Use vanishing bases for decrease and eliminate forced odd top degrees.
    The multiplier of free_face is unrestricted."""
    n = len(g_x)
    g_z = [sp.expand(gj.subs(x_in_terms_of_z)) for gj in g_x]

    if vanishing_at_xstar:
        mult_z = [poly_variable_no_low_degree(f"{tag}_{j}", z_list, d, min_deg=2)
                  for j in range(n)]
    else:
        mult_z = [poly_variable(f"{tag}_{j}", z_list, d) for j in range(n)]

    elim_subs = eliminate_forced_top_degree(mult_z, g_z, z_list, d)
    if elim_subs:
        mult_z = [sp.expand(mj.subs(elim_subs)) for mj in mult_z]

    def add_sos(e, name):
        return add_sos_constraint_scaled(prob, e, z_list, name=name,
                                         vanishing=vanishing_at_xstar,
                                         row_scaling=row_scaling)

    mult_cons = []
    for j, mj in enumerate(mult_z):
        if j == free_face:
            continue
        con_j = add_sos(mj, f"{tag}_{j}_sos")
        if mult_cap is not None:
            I = pic.Constant(np.eye(len(con_j.basis)))
            prob.add_constraint(float(mult_cap) * I - con_j.Q >> 0)
        mult_cons.append(con_j)

    expr_z = sp.expand(expr_x.subs(x_in_terms_of_z))
    target_z = sp.expand(expr_z - sum(mult_z[j] * g_z[j] for j in range(n)))
    con = add_sos(target_z, f"{tag}_target")
    return mult_z, mult_cons, con


# ===========================================================================
# (2) M's LMIs
# ===========================================================================

def add_absolute_floor(prob, cons, eps, cap=None):
    """Impose  Q >= eps*I  on every Gram matrix in `cons`, and optionally
    Q <= cap*I.

    ABSOLUTE, not relative.  Earlier versions used
    Q >= (eps/n_Q) tr(Q) I -- a bound on Q's CONDITION NUMBER, invariant
    under rescaling Q.  That has two defects which this form removes at a
    stroke:

      * it is HOMOGENEOUS, so Q = 0 satisfies it for every eps.  That is not
        hypothetical: a solve once returned sigma_dec == 0 identically, the
        multipliers alone reproducing the target, which is a valid Putinar
        certificate and useless for verification (the zero matrix is the
        corner of the cone).  A separate trace floor tr(Q) >= t_min had to be
        bolted on to exclude it.  With an absolute floor and eps > 0 the zero
        matrix is infeasible outright, so THE TRACE FLOOR IS NO LONGER
        NEEDED and is gone.
      * what rounding actually has to survive is an absolute perturbation --
        the exact projection moves lambda_min by roughly the solver's
        equality residual, a number in units of Q's entries, not a fraction
        of Q's own scale.  A relative floor therefore protects the wrong
        quantity: it buys the same relative room whether Q is large or tiny.

    `eps` may be a PICOS variable (the residual floor, which is maximised)
    or a float (the multiplier floor, which is fixed).
    """
    for con in cons:
        nQ = len(con.basis)
        I = pic.Constant(np.eye(nQ))
        prob.add_constraint(con.Q - eps * I >> 0)
        if cap is not None:
            prob.add_constraint(float(cap) * I - con.Q >> 0)


def constraint_scale_fixing_relaxed(prob, dim, M_sym, kappa, mu_val):
    """mu*I <= M <= kappa*I.

    constraint_scale_fixing_with_margin imposes tau*E + mu*I <= M <=
    kappa*I - mu*I instead.  Both extras are dropped here:

      * tau*E floors only the degree-1 block and is implied by mu*I as
        soon as mu > 0, which "maximise mu" guarantees;
      * the -mu*I discount on the cap takes room away from M for nothing
        -- the cap is a scale normalisation, not part of the Lyapunov
        conclusion.

    What remains is the whole conclusion: M >= mu*I gives
    L = Phi^T M Phi >= mu |Phi|^2 >= mu * w_1^2 * |z|^2 with w_1 the
    degree-1 weight of W, so L is positive definite in z with an explicit
    constant, and M <= kappa*I keeps "maximise mu" bounded.
    """
    M_pic = pic.block([[prob.sym_to_var(M_sym[i, j]) for j in range(dim)]
                       for i in range(dim)])
    I = pic.Constant(np.eye(dim))
    prob.add_constraint(M_pic - float(mu_val) * I >> 0)
    prob.add_constraint(float(kappa) * I - M_pic >> 0)
    return M_pic


# ===========================================================================
# (3) Decrease
# ===========================================================================

def rate_weight_needed(q):
    """Default to the p(x)-weighted decrease rate for q<2; use L for q=2."""
    return q < 2


def decrease_rate_term(x, q, L, prod_x, weighted=None):
    """The rate expression subtracted in (3), i.e. what multiplies c*r.

    It always carries the clearing factor P = prod_x that wtilde was
    multiplied by, since (3) is stated in cleared form; dividing (3) through
    by P recovers the statement about d/dt L.  On top of that it carries the
    state-dependent weight p(x) exactly when rate_weight_needed(q) says so.

        weighted=True   ->  P * p * L   certifying  d/dt L <= -c r p(x) L
        weighted=False  ->  P * L       certifying  d/dt L <= -c r L

    `weighted=None` means "decide from q", which is the intended use.
    """
    if isinstance(weighted, str):
        if weighted not in RATE_FORMS:
            raise ValueError(f"Unknown decrease rate {weighted!r}; expected {RATE_FORMS}.")
        weighted = weighted == "pL"
    if weighted is None:
        weighted = rate_weight_needed(q)
    if weighted:
        return sp.expand(prod_x * sp.prod(list(x)) * L)
    return sp.expand(prod_x * L)


def decrease_target(x, q, L, grad_L, wtilde, prod_x, xstar, rho, r, c, form):
    """(3)'s left-hand side before any multiplier is subtracted."""
    m = len(list(x))
    if not (0 < c < 2):
        raise ValueError(f"c must lie in (0,2); got c={c}")
    cr = sp.Rational(str(c)) * r
    inner = sp.expand(-sum(grad_L[i] * wtilde[i] for i in range(m)))
    rate = decrease_rate_term(x, q, L, prod_x, weighted=form)
    return sp.expand(inner - cr * rate), inner, rate


def constraint_decrease_bounded(prob, x, z, x_in_terms_of_z, m, d, g, L, grad_L,
                                wtilde, prod_x, xstar, rho, r, c, q, mu_pic,
                                mult_cap, form, row_scaling=None,
                                verbose=False):
    """Constraint (3), built with vanishing_at_xstar=True exactly with a vanishing basis:
    every chi_i and the big target live in a basis with no degree-0/1 term.

    Still NO flat margin on this target: the expression vanishes
    identically at x* (grad L(x*) = 0 by construction, wtilde(x*) = 0
    since v(x*) = 0, and every rate form is O(z^2) there) while every
    g_i(x*) > 0, so subtracting a positive constant would be infeasible by
    construction.  c*r and now the rate WEIGHT are this constraint's slack
    levers instead.

    The target has no degree-0/1 terms: each
    rate form is p(x) times something O(z^2), and chi_j is built with
    min_deg=2, so target = O(z^2) identically.
    """
    x_list = list(x)
    target, inner, rate = decrease_target(x, q, L, grad_L, wtilde, prod_x,
                                          xstar, rho, r, c, form)
    if verbose:
        print(f"    [decrease] rate form {form!r}: deg(-<grad L, wtilde>) = "
              f"{_total_degree(inner, x_list)}, "
              f"deg(rate) = {_total_degree(rate, x_list)}, "
              f"deg(target) = {_total_degree(target, x_list)}, "
              f"r={float(r):.6g}, c={c}, multiplier degree {d} "
              f"(rescaled-in-z, vanishing basis, bounded)")
    return putinar_bounded_scaled(prob, target, g, z, x_in_terms_of_z, d, "chi",
                                  mult_cap, vanishing_at_xstar=True,
                                  row_scaling=row_scaling)


# ===========================================================================
# (4) Boundary
# ===========================================================================

def boundary_normaliser(x, Hinv, m, face, boundary_norm, q):
    """nrm_face: the positive function (4) is divided through by.

    COORDINATE FACES (face < m): Hinv[face,face] = x_{face+1}^{2-q}, exactly
    the factor <grad L, Hinv grad g_face> carries, and equal to delta^{2-q}
    ON the face being certified -- which is what crushes a flat -mu margin
    This preserves the boundary sign because x_i is positive on the domain.

    SIMPLEX FACE (face == m): 1 for q <= 2.  There sum_k x_k = m forces some
    x_k >= 1, so Hinv cannot degenerate and no normalisation is needed.  For
    q > 2, though, Hinv = diag(x_i^{2-q}) has NEGATIVE exponents and
    <grad L, Hinv grad g_m> = -sum_i dL/dx_i * x_i^{2-q} is not even a
    polynomial; dividing by p(x)^{2-q} (i.e. multiplying by p^{q-2}, p =
    prod_k x_k) clears every one of those negative powers at once, since each
    term carries a negative power of a single distinct x_i.

    Every branch returns a function that is strictly POSITIVE on X, which is
    all the sign conclusion of (4) needs: the certified statement becomes
    <grad L, Hinv grad g_face> <= -mu * nrm_face < 0 on that face.
    """
    if not boundary_norm:
        return sp.Integer(1)
    if face < m:
        return Hinv[face, face]
    if q <= 2:
        return sp.Integer(1)
    return sp.prod(x) ** (2 - q)


def boundary_inner_product(x, m, g, grad_L, Hinv, face, boundary_norm, q):
    """<grad L, Hinv grad g_face>, divided by nrm_face.

    The division is exact: Hinv is diagonal and grad g_face = e_face for
    face < m, so the quotient is literally dL/dx_{face+1}.  cancel() is
    used rather than that shortcut so the code stays correct if Hinv ever
    stops being diagonal; the assertion below is what guarantees the
    result is still a polynomial.
    """
    x_list = list(x)
    dg = [sp.diff(g[face], xi) for xi in x_list]
    ip = sp.expand(sum(grad_L[i] * Hinv[i, j] * dg[j]
                       for i in range(m) for j in range(m)))
    nrm = boundary_normaliser(x, Hinv, m, face, boundary_norm, q)
    if nrm == 1:
        if not ip.is_polynomial(*x_list):
            raise ValueError(
                f"boundary_inner_product: face {face} is not polynomial at q={q} "
                f"and boundary_norm is off; turn it on (it is what clears the "
                f"negative powers the metric contributes for q > 2).")
        return ip
    ip = sp.expand(sp.cancel(ip / nrm))
    if not ip.is_polynomial(*x_list):               # pragma: no cover
        raise ValueError(
            f"boundary_inner_product: face {face}'s quotient by {nrm} is not "
            f"polynomial -- boundary_norm is only valid for a diagonal Hinv.")
    return ip


def constraint_boundary_bounded(prob, x, z, x_in_terms_of_z, m, d, g, grad_L,
                                Hinv, face, boundary_norm, q, mu_sym, mu_pic,
                                mult_cap, row_scaling=None, verbose=False):
    """Constraint (4) on one face, with the metric-normalised strict margin

        -<grad L, Hinv grad g_face>/nrm_face - mu - varphi_face*g_face
            - sum_{j != face} eta_{face,j}*g_j   in SOS,

    i.e. <grad L, Hinv grad g_face> <= -mu*nrm_face < 0 on the face.  With
    boundary_norm=False this is flat -mu.

    Subtracting a flat mu is safe here, unlike Decrease: the inner product
    has no reason to vanish anywhere, so there is no analogue of
    Decrease's x* obstruction.  vanishing_at_xstar is deliberately NOT set
    -- this target has -mu as a genuine constant term.
    """
    ip = boundary_inner_product(x, m, g, grad_L, Hinv, face, boundary_norm, q)
    nrm = boundary_normaliser(x, Hinv, m, face, boundary_norm, q)
    if verbose:
        note = f"normalised by {nrm}" if nrm != 1 else "unnormalised"
        print(f"    [boundary face={face}] deg = {_total_degree(ip, list(x))} "
              f"({note}), multiplier degree {d}, margin -mu")
    target = sp.expand(-ip - mu_sym)
    return putinar_bounded_scaled(prob, target, g, z, x_in_terms_of_z, d, f"s{face}",
                                  mult_cap, free_face=face,
                                  row_scaling=row_scaling)


# ===========================================================================
# Face-reduced boundary certificates
# ===========================================================================

def face_parameterisation(x, m, lower_bound, face):
    r"""Affine chart x = T y + b of face i, with y in the standard simplex.

    Face $i<m$ is $\{x_{i+1}=\delta,\ x_k\ge\delta,\ \sum_k x_k\le m\}$.
    Substituting $x_k=\delta+t_k$ leaves $t_k\ge0$ and
    $\sum_{k\ne i}t_k\le m-\delta-(m-1)\delta = m(1-\delta)$, an
    $(m-1)$-simplex.  Face $m$ is $\{\sum_k x_k=m,\ x_k\ge\delta\}$, where
    the same substitution gives $\sum_k t_k = m(1-\delta)$ exactly -- again an
    $(m-1)$-simplex, with the last coordinate solved for.

    Scaling by $S=m(1-\delta)$ puts both on the STANDARD simplex
    $\Delta_{m-1}=\{y: y_k\ge0,\ 1-\sum y_k\ge0\}$, whose $m$ defining
    forms are returned as `ell`.  Everything is exact rational.

    Returns (subs, y, ell): `subs` maps each $x_k$ to its affine expression in
    $y$, `y` the $m-1$ chart variables, `ell` the $m$ simplex forms.
    """
    d = sp.Rational(lower_bound)
    S = sp.Integer(m) * (1 - d)
    y = sp.symbols(f"y1:{m}", real=True) if m > 1 else ()
    y = tuple(y)
    ell = [yi for yi in y] + [sp.Integer(1) - sum(y)]

    subs = {}
    if face < m:
        subs[x[face]] = d
        others = [k for k in range(m) if k != face]
        for j, k in enumerate(others):
            subs[x[k]] = d + S * y[j]
    else:
        for j in range(m - 1):
            subs[x[j]] = d + S * y[j]
        subs[x[m - 1]] = d + S * (sp.Integer(1) - sum(y))
    return subs, y, ell


def constraint_boundary_face_reduced(prob, x, m, d, grad_L, Hinv, face, q,
                                     lower_bound, mu_val, mult_cap,
                                     row_scaling=None, verbose=False):
    r"""The face-reduced form of the boundary certificate.

        B_i(T_i y + b_i) - mu - sum_k chi_{i,k}(y) ell_{i,k}(y) = sigma_i(y),
        chi_{i,k}, sigma_i in SOS[y],

    with B_i the metric-normalised inward quantity of
    boundary_inner_product.  Because $g_i=0$ is imposed by SUBSTITUTION
    rather than by a multiplier, the free ambient multiplier $\varphi_i g_i$
    disappears entirely, and the certificate lives in $m-1$ variables instead
    of $m$ -- for $m=2$ that is a univariate SOS problem, whose Gram matrices
    are a few rows square instead of six to ten.

    The conclusion is identical: $B_i\ge\mu>0$ on the face, hence
    $\langle\nabla L,H^{-1}\nabla g_i\rangle\le-\mu\nu_i<0$ there.
    """
    g_amb = define_constraints(x, m, lower_bound)
    B = boundary_inner_product(x, m, g_amb, grad_L, Hinv, face, True, q)
    B = sp.expand(-B)                     # boundary_inner_product returns ip
    subs, y, ell = face_parameterisation(x, m, lower_bound, face)
    B_y = sp.expand(B.subs(subs))
    y_list = list(y)

    mults = [poly_variable(f"chi{face}_{k}", y_list, d) for k in range(len(ell))]
    elim = eliminate_forced_top_degree(mults, ell, y_list, d)
    if elim:
        mults = [sp.expand(mj.subs(elim)) for mj in mults]

    mult_cons = []
    for k, mk in enumerate(mults):
        con_k = add_sos_constraint_scaled(prob, mk, y_list,
                                          name=f"chi{face}_{k}_sos",
                                          vanishing=False, row_scaling=row_scaling)
        if mult_cap is not None:
            I = pic.Constant(np.eye(len(con_k.basis)))
            prob.add_constraint(float(mult_cap) * I - con_k.Q >> 0)
        mult_cons.append(con_k)

    target = sp.expand(B_y - sp.Rational(str(mu_val))
                       - sum(mults[k] * ell[k] for k in range(len(ell))))
    if verbose:
        print(f"    [boundary face={face}, FACE-REDUCED] deg(B) = "
              f"{_total_degree(B_y, y_list) if y_list else 0} in "
              f"{len(y_list)} var(s), multiplier degree {d}, "
              f"margin -mu={float(mu_val):g}")
    con = add_sos_constraint_scaled(prob, target, y_list, name=f"sigma{face}",
                                    vanishing=False, row_scaling=row_scaling)
    return mults, mult_cons, con


def boundary_face_degrees(s, l, n, m, d_L, q, lower_bound=DELTA, rho=None,
                          lyap_basis="full", basis_rescale="relative"):
    """Degree of the face-reduced target B_i(T_i y + b_i), one per face."""
    x, v, xstar, rho_val, Hinv, L, grad_L = _degree_setup(
        s, l, n, m, d_L, q, lower_bound, rho, lyap_basis, basis_rescale)
    g = define_constraints(x, m, lower_bound)
    out = []
    for face in range(m + 1):
        B = boundary_inner_product(x, m, g, grad_L, Hinv, face, True, q)
        subs, y, ell = face_parameterisation(x, m, lower_bound, face)
        By = sp.expand(sp.expand(B).subs(subs))
        out.append(_total_degree(By, list(y)) if y else 0)
    return out


# ===========================================================================
# Degree bookkeeping accounts for the rate form and boundary normalization.
# ===========================================================================

def _degree_setup(s, l, n, m, d_L, q, lower_bound, rho, lyap_basis, basis_rescale):
    x, v, xstar, _ = build_game(s, l, n, m)
    rho_val = domain_scale(xstar, lower_bound, rho)
    Hinv = build_metric_matrix(s, l, n, m, "q", q=q)
    monoms = lyapunov_monomials(m, d_L, lyap_basis)
    weights = basis_weights(monoms,
                            z_amplitude(m, xstar, rho_val, lower_bound),
                            metric_amplitude(q, m, lower_bound),
                            basis_rescale)
    L, grad_L, M_sym, _, _, _ = build_lyapunov(
        x, m, d_L, xstar, rho_val, lyap_basis=lyap_basis, weights=weights)
    return x, v, xstar, rho_val, Hinv, L, grad_L


def decrease_intrinsic_degree(s, l, n, m, d_L, q, weighted=None,
                              lower_bound=DELTA, rho=None,
                              lyap_basis="full", basis_rescale="relative"):
    """Degree of (3)'s left-hand side, before any multiplier."""
    x, v, xstar, rho_val, Hinv, L, grad_L = _degree_setup(
        s, l, n, m, d_L, q, lower_bound, rho, lyap_basis, basis_rescale)
    wtilde, prod_x = scaled_dynamics_poly_q(x, v, Hinv, q)
    inner = sp.expand(-sum(grad_L[i] * wtilde[i] for i in range(m)))
    rate = decrease_rate_term(x, q, L, prod_x, weighted=weighted)
    return _total_degree(sp.expand(inner - rate), list(x))


def boundary_intrinsic_degrees(s, l, n, m, d_L, q, boundary_norm=True,
                               lower_bound=DELTA, rho=None,
                               lyap_basis="full", basis_rescale="relative"):
    """Degree of (4)'s inner product, ONE PER FACE.

    This need not be the same number for every face once
    boundary_norm is on: the m coordinate faces lose 2-q degrees to the
    normalisation, the simplex face keeps its own.  Returning the list
    rather than the max is what lets each face carry its own multiplier
    degree -- for q=0, d_L=4 that is degree 4 on the three coordinate
    faces against 6 on the simplex face, i.e. a materially smaller SDP
    than one shared d_bnd = 6.
    """
    x, v, xstar, rho_val, Hinv, L, grad_L = _degree_setup(
        s, l, n, m, d_L, q, lower_bound, rho, lyap_basis, basis_rescale)
    g = define_constraints(x, m, lower_bound)
    return [_total_degree(
        boundary_inner_product(x, m, g, grad_L, Hinv, face, boundary_norm, q), list(x))
        for face in range(m + 1)]


# ===========================================================================
# Build / solve
# ===========================================================================

def build_sdp_problem_v2(s, l, n, m, d_L, d_dec, d_bnd, q,
                         c=C_RATE, kappa=KAPPA, mult_cap=KAPPA,
                         lower_bound=DELTA, rho=None,
                         boundary_mode="face", decrease_rate=None,

                         r_max_den=MAX_DEN,
                         mu=MU_FIXED, eps_M=EPS_M,
                         row_scaling=None,
                         verbose=False):
    """Assemble the fixed-mu SDP maximizing the absolute residual floor eps_R.

    Return the problem and symbolic data needed for exact verification.
    decrease_rate selects L or pL; None selects from q."""
    x, v, xstar, _ = build_game(s, l, n, m)
    rho_val = domain_scale(xstar, lower_bound, rho)
    z, x_in_terms_of_z = build_z(x, xstar, rho_val)

    g = define_constraints(x, m, lower_bound)
    Hinv = build_metric_matrix(s, l, n, m, "q", q=q)
    wtilde, prod_x = scaled_dynamics_poly_q(x, v, Hinv, q)
    r, alpha = spectral_rate(x, v, xstar, Hinv, max_den=r_max_den)

    zmax = z_amplitude(m, xstar, rho_val, lower_bound)
    hmax = metric_amplitude(q, m, lower_bound)
    monoms = lyapunov_monomials(m, d_L, "full")
    weights = basis_weights(monoms, zmax, hmax, "relative")
    if verbose:
        print(f"  BNE x* = {xstar}, rho = {rho_val}  (z = (x-x*)/rho)")
        print(f"  alpha(A) = {alpha:.6g}  =>  r = {float(r):.6g} "
              f"(rationalised, A = Hinv(x*) @ Dv(x*))")
        if True:
            print(f"  [basis rescale] Zmax = {float(zmax):.4g}, Hmax = {float(hmax):.4g}; "
                  f"W = diag({', '.join(f'{float(w):.4g}' for w in weights)})")

    prob = BertrandSOSProblem()

    # mu is a FIXED constant now, not a maximised variable.  What is maximised
    # instead is eps_R, the ABSOLUTE floor on the residual Gram matrices --
    # "give me a certificate whose Gram matrices are as far inside the cone as
    # possible", rather than "give me the steepest Lyapunov function".  The two
    # objectives are genuinely different: mu measures how strongly L grows away
    # from x*, eps_R measures how much room the exact rounding has.  Only the
    # second is what verification needs.
    eps_R = pic.RealVariable("eps_R")
    prob.add_constraint(eps_R >= 0)
    prob.add_constraint(eps_R <= kappa)

    L, grad_L, M_sym, monoms, weights, dim = build_lyapunov(
        x, m, d_L, xstar, rho_val, lyap_basis="full", weights=weights)
    constraint_scale_fixing_relaxed(prob, dim, M_sym, kappa, mu)
    if verbose:
        w1 = min(float(w) for w, a in zip(weights, monoms) if sum(a) == 1)
        print(f"    [scale-fixing] deg(L) = {_total_degree(L, list(x))}, "
              f"M is {dim}x{dim}; mu fixed at {float(mu):g}, so "
              f"L >= {float(mu) * w1 ** 2:.4g}*|z|^2")

    bounded_mult_cons = []
    big_target_cons = []

    _, dec_mult_cons, dec_con = constraint_decrease_bounded(
        prob, x, z, x_in_terms_of_z, m, d_dec, g, L, grad_L, wtilde, prod_x,
        xstar, rho_val, r, c, q, None, mult_cap, decrease_rate,
        row_scaling=row_scaling, verbose=verbose)
    bounded_mult_cons.extend(dec_mult_cons)
    big_target_cons.append(dec_con)

    d_bnd_list = ([d_bnd] * (m + 1) if isinstance(d_bnd, int) else list(d_bnd))
    if verbose:
        d_note = d_bnd_list[0] if len(set(d_bnd_list)) == 1 else d_bnd_list
        print(f"    [boundary FACE-REDUCED] {m + 1} faces, {m - 1} var(s), "
              f"multiplier degree(s) {d_note}, margin -mu={float(mu):g}")
    for face in range(m + 1):
        if boundary_mode == "face":
            _, bm, bc = constraint_boundary_face_reduced(
                prob, x, m, d_bnd_list[face], grad_L, Hinv, face, q,
                lower_bound, mu, mult_cap, row_scaling=row_scaling,
                verbose=False)
        elif boundary_mode == "ambient":
            _, bm, bc = constraint_boundary_bounded(
                prob, x, z, x_in_terms_of_z, m, d_bnd_list[face], g, grad_L,
                Hinv, face, True, q, sp.Rational(str(mu)), None, mult_cap,
                row_scaling=row_scaling, verbose=False)
        else:
            raise ValueError(
                f"boundary_mode must be 'face' or 'ambient'; got {boundary_mode!r}.")
        bounded_mult_cons.extend(bm)
        big_target_cons.append(bc)

    # Absolute floors.  Multipliers get a FIXED eps_M; residuals get the
    # maximised eps_R.  No trace floor: an absolute floor already excludes the
    # zero matrix (see add_absolute_floor).
    add_absolute_floor(prob, bounded_mult_cons, float(eps_M), cap=mult_cap)
    add_absolute_floor(prob, big_target_cons, eps_R, cap=None)
    if verbose:
        print(f"    [floors] Q >= {float(eps_M):g}*I on {len(bounded_mult_cons)} "
              f"multipliers (fixed), Q >= eps_R*I on {len(big_target_cons)} "
              f"residuals (MAXIMISED)")

    return (prob, L, x, xstar, M_sym, dim, r, alpha, mu, eps_R, rho_val,
            weights, monoms, bounded_mult_cons, big_target_cons)


@dataclass
class MarginCertificateResultV2:
    s: int
    l: int
    n: int
    m: int
    q: int
    d_L: int
    d_dec: int
    d_bnd: str
    n_sdp: int
    boundary_mode: str
    c: float
    delta: str
    eps_R: float
    eps_M: float
    rho: float
    alpha: float
    r: float
    mu: float
    w1: float
    mu_z2: float
    status: str
    runtime_s: float
    solver: str
    solver_status: str
    L: Optional[str] = field(default=None, repr=False)

    def row(self):
        d = asdict(self)
        d.pop("L")
        return d


CSV_FIELDS = [f for f in MarginCertificateResultV2.__dataclass_fields__ if f != "L"]


def solve_sdp_v2(s, l, n, m, d_L, q,
                 d_dec=None, d_bnd=None, d_dec_bump=2, d_bnd_bump=2,
                 c=C_RATE, kappa=KAPPA, mult_cap=KAPPA,
                 lower_bound=DELTA, rho=None,
                 boundary_mode="face", decrease_rate=None,
                 mu=MU_FIXED, eps_M=EPS_M,
                 row_scaling=None,
                 solver="mosek", timelimit=900,
                 solve_options=None,
                 fallback_solver=None,
                 do_exact_verify=False,
                 max_den=MAX_DEN, q_max_den=Q_MAX_DEN,
                 screen_tol=1e-9, mu_buffer_frac=0.9,
                 psd_fallback=True,
                 verify_progress=True,
                 verify_max_basis=30,
                 _report_sink=None,
                 verbose=True):
    """Build, solve and (optionally) exactly verify one configuration.

    d_dec/d_bnd default to their natural floor for THIS file's degrees
    (which depend on q, since the rate weight does); raising them past it
    buys margin at the cost of problem size.  solve_options is forwarded to
    PICOS' Problem.solve, allowing callers to tighten MOSEK tolerances.
    fallback_solver defaults to None so MOSEK failures are reported directly
    instead of silently rerunning the problem through CVXOPT.  psd_fallback
    controls whether exact verification pays for SymPy's rank-deficient PSD
    prover after direct projection lands near the cone boundary.
    verify_progress controls compact progress lines for exact Gram checks
    independently of verbose solver/build output.  Everything else is
    documented in the module docstring.
    """
    if q not in (0, 1, 2):
        raise ValueError(f"q must be in {{0,1,2}}; got {q}.")
    if d_bnd is None and boundary_mode == "face":
        d_bnd = [multiplier_degree_floor(di) + d_bnd_bump
                 for di in boundary_face_degrees(
                     s, l, n, m, d_L, q, lower_bound=lower_bound, rho=rho)]
    elif d_bnd is None:
        d_bnd = [multiplier_degree_floor(di) + d_bnd_bump
                 for di in boundary_intrinsic_degrees(
                     s, l, n, m, d_L, q, boundary_norm=True,
                     lower_bound=lower_bound, rho=rho)]
    if d_dec is None:
        d_dec = multiplier_degree_floor(decrease_intrinsic_degree(
            s, l, n, m, d_L, q, weighted=decrease_rate,
            lower_bound=lower_bound, rho=rho)) + d_dec_bump
    t0 = time.perf_counter()

    if verbose:
        print("=" * 72)
        print(f"  Game        : s={s} ({DEMAND_NAMES.get(s, s)}), l={l}, n={n}, m={m}")
        print(f"  Regularizer : q={q} ({Q_NAMES.get(q, q)})")
        print(f"  Degrees     : d_L={d_L}, d_dec={d_dec}, d_bnd(per face)={d_bnd}")
        print(f"  Boundary    : {boundary_mode!r}")
        print(f"  Decrease    : rate weight {decrease_rate or ('pL' if rate_weight_needed(q) else 'L')}")
        print(f"  Constants   : delta={lower_bound}, kappa={kappa:g}, c={c:g}, "
              f"mu={float(mu):g} (FIXED), eps_M={float(eps_M):g} (FIXED), "
              f"max_den={max_den:g}")
        print(f"  Objective   : MAXIMISE eps_R, the absolute floor Q >= eps_R*I "
              f"on every residual Gram matrix")
        print("=" * 72)

    try:
        (prob, L, x, xstar, M_sym, dim, r, alpha, mu, eps_R, rho_val,
         weights, monoms, bounded_mult_cons, big_target_cons) = build_sdp_problem_v2(
            s, l, n, m, d_L, d_dec, d_bnd, q, c=c, kappa=kappa,
            mult_cap=mult_cap, lower_bound=lower_bound, rho=rho,
            boundary_mode=boundary_mode, decrease_rate=decrease_rate,
            r_max_den=max_den,
            mu=mu, eps_M=eps_M, row_scaling=row_scaling, verbose=verbose)
    except RuntimeError as exc:
        if verbose:
            print(f"  ABORT: {exc}")
        return MarginCertificateResultV2(
            s, l, n, m, q, d_L, d_dec, str(d_bnd), 0, boundary_mode, c,
            str(lower_bound), None, float(eps_M),
            float("nan"), float("nan"), float("nan"), float(mu), float("nan"),
            float("nan"), "not_hurwitz",
            time.perf_counter() - t0, solver, "not_attempted", L=None)

    w1 = min(float(w) for w, a in zip(weights, monoms) if sum(a) == 1)

    n_sdp = sdp_size(prob)
    if verbose:
        print(f"  SDP size    : {n_sdp} scalar variables")
        opt_note = f", solve_options={solve_options}" if solve_options else ""
        print(f"  Solving (maximizing residual floor eps_R) with '{solver}'{opt_note} ...",
              flush=True)

    feasible, solver_used, detail = _solve_max_margin(prob, eps_R, solver,
                                                      timelimit=timelimit,
                                                      solve_options=solve_options,
                                                      fallback_solver=fallback_solver)
    eps_R_val = None
    L_val = None
    if feasible and detail in ("optimal", "feasible"):
        status = "certified"
        try:
            eps_R_val = max(0.0, float(eps_R.value))
        except Exception:
            eps_R_val = None
        L_val = prob.subs_with_sol(L)
    elif feasible:
        status = "no_solution"
    elif "infeasible" in detail.lower():
        status = "infeasible"
    else:
        status = "no_solution"

    if verbose:
        print(f"\n  solver status: {detail!r} -> {status}")
        if status == "certified":
            e_note = f"{eps_R_val:.4g}" if eps_R_val is not None else "?"
            note = ("attempting exact verification" if do_exact_verify
                    else "exact verification skipped (do_exact_verify=False)")
            print(f"  FEASIBLE: certificate found, residual floor eps_R={e_note} "
                  f"(Q >= eps_R*I on every sigma), with "
                  f"L(x) >= {float(mu) * w1 ** 2:.4g}*|z|^2 -- {note}.")
        elif status == "infeasible":
            print("  INFEASIBLE at this shape -- raise d_dec (then d_L, d_bnd), "
                  "loosen mult_cap, or lower c, before concluding anything "
                  "about the dynamics.")
        else:
            print(f"  INCONCLUSIVE (solver status {detail!r}) -- mosek is not "
                  f"run-to-run reproducible here; retry before believing it.")

    biggest = max((len(con.basis) for con in big_target_cons + bounded_mult_cons),
                  default=0)
    if (do_exact_verify and status == "certified" and verify_max_basis
            and biggest > verify_max_basis):
        # The exact pass is dominated by rational LDL on the largest Gram
        # block, whose cost grows steeply in its dimension (a 20x20 block
        # takes ~2 minutes; a 44x44 one, which d_dec=14 produces, did not
        # finish in hours).  Skipping it is reported as its own status
        # rather than as a verification FAILURE -- nothing was disproved,
        # the check was not run.
        if verbose:
            print(f"\n  EXACT VERIFICATION SKIPPED: largest Gram basis is "
                  f"{biggest} > verify_max_basis={verify_max_basis}; the exact "
                  f"pass is not tractable at this size. Stage 1 stands.")
        status = "certified_verify_skipped"
    elif do_exact_verify and status == "certified":
        report = verify_certificate_exact_v2(
            prob, L, x, xstar, M_sym, mu, bounded_mult_cons, big_target_cons,
            dim, kappa, mult_cap, weights, monoms, m_lmi="relaxed",
            max_den=max_den, q_max_den=q_max_den, screen_tol=screen_tol,
            mu_buffer_frac=mu_buffer_frac, psd_fallback=psd_fallback,
            verbose=verify_progress)
        if _report_sink is not None:
            _report_sink["report"] = report
        if verbose and not verify_progress:
            print_exact_verification_v2(report)
        if not report["ok"]:
            status = "certified_not_verified"

    return MarginCertificateResultV2(
        s, l, n, m, q, d_L, d_dec, str(d_bnd), n_sdp, boundary_mode, c,
        str(lower_bound), eps_R_val, float(eps_M),
        float(rho_val), alpha, float(r), float(mu), w1,
        float(mu) * w1 ** 2, status,
        time.perf_counter() - t0, solver_used, detail,
        L=None if L_val is None else str(L_val))


# ===========================================================================
# Exact rational verification -- Peyrl-Parrilo, with a vanishing basis.  The only change
# is (2)'s check: the LOWER LMI alone gates the verdict.
# ===========================================================================

def _numeric_min_eig(Q_exact):
    """float lambda_min of an exact rational matrix -- a SCREEN, never a proof.

    Converting to float loses exactness by construction; this value is only
    ever used to decide WHICH exact test to run, never to answer whether a
    matrix is PSD.  See direct_gram_matrix for why that is sound.
    """
    n = Q_exact.rows
    A = np.array([[float(Q_exact[i, j]) for j in range(n)] for i in range(n)],
                 dtype=float)
    A = (A + A.T) / 2.0
    return float(np.linalg.eigvalsh(A).min())


def direct_gram_matrix(con, rational_vals, q_max_den=Q_MAX_DEN, screen_tol=1e-9,
                       psd_fallback=True):
    """Round a Gram matrix, project onto exact coefficient identities, and test PSD.

    A negative floating eigenvalue screen rejects the candidate. Positive
    results require an exact PSD test; borderline matrices require the
    optional exact semidefinite fallback. No second SDP is solved."""
    Q_final, identity_ok = exact_gram_matrix(con, rational_vals, q_max_den)
    lam = _numeric_min_eig(Q_final)
    if screen_tol > 0 and lam <= screen_tol and not psd_fallback:
        psd_ok = False
        psd_reason = (f"not accepted by direct projection screen "
                      f"(lambda_min={lam:.3g} <= screen_tol={screen_tol:.3g}; "
                      f"exact PSD fallback disabled)")
    elif screen_tol > 0 and lam < -screen_tol:
        psd_ok = False
        psd_reason = (f"not PSD (numerical screen, lambda_min={lam:.3g}; "
                      f"exact test skipped -- see direct_gram_matrix)")
    else:
        psd_ok, psd_reason = is_psd_exact(Q_final)
    return Q_final, identity_ok, psd_ok, psd_reason, {
        "method": "direct", "t_star": None, "lambda_min_float": lam}


def direct_verify_bounded_mult(con, rational_vals, mult_cap,
                               q_max_den=Q_MAX_DEN, screen_tol=1e-9,
                               psd_fallback=True):
    """Exact verification of ONE bounded multiplier by direct projection.

    Both LMIs putinar_bounded imposed on the block are checked against the
    SAME projected Q -- Q >> 0 (SOS-ness) and mult_cap*I - Q >> 0 (5).  The
    cap check stays fully exact with no screen: it is cheap, and it is a
    POSITIVE verdict, which the screen has no business short-cutting.
    """
    size = len(con.basis)
    try:
        Q_final, identity_ok, psd_ok, psd_reason, info = direct_gram_matrix(
            con, rational_vals, q_max_den=q_max_den, screen_tol=screen_tol,
            psd_fallback=psd_fallback)
        if mult_cap is not None:
            cap_exact = rationalize_scalar(mult_cap, q_max_den)
            cap_ok, cap_reason = is_psd_exact(cap_exact * sp.eye(size) - Q_final)
        else:
            cap_ok, cap_reason = True, "no cap"
    except Exception as exc:
        identity_ok = psd_ok = cap_ok = False
        psd_reason = cap_reason = f"error: {exc!r}"
        info = {"method": "error", "t_star": None}
    ok = identity_ok and psd_ok and cap_ok
    return {"size": size, "identity_ok": identity_ok,
            "psd_ok": psd_ok, "psd_reason": psd_reason,
            "cap_ok": cap_ok, "cap_reason": cap_reason, "ok": ok,
            "recovery": info}


def exact_verify_M_v2(M_sym, rational_vals, dim, kappa, mu_sym, m_lmi,
                      m=None, tau=TAU, max_den=MAX_DEN, mu_buffer_frac=0.9):
    """Exact PSD check of (2).

    m_lmi="relaxed":  lower  M - mu_buf*I >> 0          <- gates the verdict
                      upper  kappa*I - M >> 0           <- informational only
    m_lmi="strict":    M - tau*E - mu_buf*I >> 0  and
                            kappa*I - M - mu_buf*I >> 0, both gating.

    No re-projection is needed: M's entries ARE the free SDP variables the
    constraint tests.  mu_buffer_frac < 1 checks against a FRACTION of the
    rounded mu -- (2)'s lower LMI is exactly the one whose slack IS the
    maximised mu, so it is tight at the optimum by construction and leaves
    nothing for rounding.  Discounting is logically free (M >> mu*I implies
    M >> frac*mu*I for frac in [0,1]); it costs only a smaller REPORTED
    margin.

    The upper LMI does not gate in "relaxed" mode because it is a scale
    NORMALISATION of L, not part of the Lyapunov conclusion: L still
    certifies stability whatever its overall size.
    """
    M_exact = sp.Matrix(dim, dim, lambda i, j: rational_vals[M_sym[i, j]])
    kappa_exact = rationalize_scalar(kappa, max_den)
    # mu is a FIXED constant in this program, not an SDP variable, so it is
    # not in rational_vals -- rationalise it directly.  (It is also no longer
    # tight at the optimum, so the buffer below is belt-and-braces.)
    mu_exact = (rational_vals[mu_sym] if mu_sym in rational_vals
                else sp.Rational(str(mu_sym)))
    mu_buf = mu_exact * sp.Rational(str(mu_buffer_frac))
    if m_lmi == "strict":
        E = sp.diag(*([sp.Integer(1)] * m + [sp.Integer(0)] * (dim - m)))
        lower = M_exact - sp.Rational(str(tau)) * E - mu_buf * sp.eye(dim)
        upper = kappa_exact * sp.eye(dim) - M_exact - mu_buf * sp.eye(dim)
        upper_gates = True
    else:
        lower = M_exact - mu_buf * sp.eye(dim)
        upper = kappa_exact * sp.eye(dim) - M_exact
        upper_gates = False
    lower_ok, lower_reason = is_psd_exact(lower)
    upper_ok, upper_reason = is_psd_exact(upper)
    return {"lower_ok": lower_ok, "lower_reason": lower_reason,
            "upper_ok": upper_ok, "upper_reason": upper_reason,
            "upper_gates": upper_gates,
            "mu_exact": mu_exact, "mu_buffered": mu_buf}


def verify_certificate_exact_v2(prob, L, x, xstar, M_sym, mu,
                                bounded_mult_cons, big_target_cons,
                                dim, kappa, mult_cap, weights, monoms,
                                m_lmi="relaxed", m=None,
                                max_den=MAX_DEN, q_max_den=Q_MAX_DEN,
                                screen_tol=1e-9, mu_buffer_frac=0.9,
                                psd_fallback=True, verbose=True):
    """Full exact-verification pass for one solved certificate.

    report["ok"] is True iff every identity holds exactly over Q and every
    gating LMI / Gram matrix is exactly on the right side of its cone.  W
    needs no un-scaling step: its entries are exact rationals folded into
    L's coefficients when the candidate was built, so rationalising M's
    entries is already the un-scaled statement.
    """
    if m is None:
        m = len(list(x))
    rational_vals = {sym: rationalize_scalar(var.value, max_den)
                     for sym, var in prob._sym_var_map.items()
                     if var.value is not None}

    try:
        L_exact, rational_vals, L_identity_ok = exact_rationalize_L(
            L, x, xstar, rational_vals, max_den)
    except Exception as exc:
        return {"ok": False, "reason": f"could not rationalise L exactly: {exc!r}"}

    M_report = exact_verify_M_v2(M_sym, rational_vals, dim, kappa, mu, m_lmi,
                                 m=m, max_den=max_den,
                                 mu_buffer_frac=mu_buffer_frac)
    if verbose:
        gate = "gating" if M_report["upper_gates"] else "info"
        print(f"\nExact verification: M lower="
              f"{'OK' if M_report['lower_ok'] else 'FAIL'} "
              f"(mu_buf={float(M_report['mu_buffered']):.4g}), "
              f"upper={'OK' if M_report['upper_ok'] else 'FAIL'} ({gate})")

    mult_reports = []
    all_mult_ok = True
    if verbose:
        print(f"Exact verification: multiplier Grams 0/{len(bounded_mult_cons)}",
              flush=True)
    for k, con in enumerate(bounded_mult_cons):
        rep = direct_verify_bounded_mult(con, rational_vals, mult_cap, q_max_den,
                                         screen_tol=screen_tol,
                                         psd_fallback=psd_fallback)
        all_mult_ok = all_mult_ok and rep["ok"]
        mult_reports.append(rep)
        if verbose:
            print(f"  multiplier Grams {k + 1}/{len(bounded_mult_cons)}: "
                  f"{'OK' if rep['ok'] else 'FAIL'}", flush=True)

    gram_reports = []
    all_gram_ok = True
    if verbose:
        print(f"Exact verification: residual Grams 0/{len(big_target_cons)}",
              flush=True)
    for k, con in enumerate(big_target_cons):
        size = len(con.basis)
        try:
            Q_final, identity_ok, psd_ok, psd_reason, rec_info = direct_gram_matrix(
                con, rational_vals, q_max_den=q_max_den, screen_tol=screen_tol,
                psd_fallback=psd_fallback)
        except Exception as exc:
            identity_ok, psd_ok, psd_reason = False, False, f"error: {exc!r}"
            rec_info = {"method": "error", "t_star": None}
        ok = identity_ok and psd_ok
        all_gram_ok = all_gram_ok and ok
        gram_reports.append({"size": size, "identity_ok": identity_ok,
                             "psd_ok": psd_ok, "psd_reason": psd_reason,
                             "recovery": rec_info})
        if verbose:
            print(f"  residual Grams {k + 1}/{len(big_target_cons)}: "
                  f"{'OK' if ok else 'FAIL'}", flush=True)

    if verbose:
        mode = "PSD fallback" if psd_fallback else "direct screen"
        n_mult = len(mult_reports)
        n_mult_ok = sum(1 for rep in mult_reports if rep["ok"])
        n_gram = len(gram_reports)
        n_gram_ok = sum(1 for rep in gram_reports
                        if rep["identity_ok"] and rep["psd_ok"])
        print(f"Exact verification: multipliers {n_mult_ok}/{n_mult} OK, "
              f"residual Grams {n_gram_ok}/{n_gram} OK ({mode}).")
        for label, reports in (("mult", mult_reports), ("gram", gram_reports)):
            for k, rep in enumerate(reports):
                ok = (rep["ok"] if label == "mult"
                      else rep["identity_ok"] and rep["psd_ok"])
                if ok:
                    continue
                reason = rep["psd_reason"]
                if label == "mult" and not rep["cap_ok"]:
                    reason = f"{reason}; cap {rep['cap_reason']}"
                print(f"  {label}[{k + 1}] size {rep['size']}x{rep['size']}: "
                      f"identity={'OK' if rep['identity_ok'] else 'FAIL'}, "
                      f"psd={'OK' if rep['psd_ok'] else 'FAIL'} ({reason})")

    m_ok = M_report["lower_ok"] and (M_report["upper_ok"] or not M_report["upper_gates"])
    ok = L_identity_ok and m_ok and all_mult_ok and all_gram_ok
    return {
        "ok": ok,
        "L_exact": sp.expand(L_exact),
        "L_identity_ok": L_identity_ok,
        "M_report": M_report,
        "mult_reports": mult_reports,
        "gram_reports": gram_reports,
        "psd_fallback": psd_fallback,
    }


def print_exact_verification_v2(report):
    if "L_identity_ok" not in report:
        print(f"\nEXACT VERIFICATION: not attempted ({report.get('reason', 'unknown')}).")
        return
    M = report["M_report"]
    m_ok = M["lower_ok"] and (M["upper_ok"] or not M["upper_gates"])
    n_mult = len(report["mult_reports"])
    n_mult_fail = sum(1 for rep in report["mult_reports"] if not rep["ok"])
    n_mult_ok = n_mult - n_mult_fail
    n_gram = len(report["gram_reports"])
    n_fail = sum(1 for g in report["gram_reports"]
                 if not (g["identity_ok"] and g["psd_ok"]))
    n_gram_ok = n_gram - n_fail
    mode = "PSD fallback" if report.get("psd_fallback") else "direct screen"
    verdict = "CERTIFIED" if report["ok"] else "FAILED"
    print(f"\nEXACT VERIFICATION: {verdict} ({mode})")
    print(f"  L identity={report['L_identity_ok']}; M lower={M['lower_ok']}; "
          f"multipliers={n_mult_ok}/{n_mult}; residual Grams={n_gram_ok}/{n_gram}")

    failures = []
    for k, rep in enumerate(report["mult_reports"], start=1):
        if rep["ok"]:
            continue
        reason = rep["psd_reason"]
        if not rep["cap_ok"]:
            reason = f"{reason}; cap {rep['cap_reason']}"
        failures.append(("mult", k, rep["size"], rep["identity_ok"],
                         rep["psd_ok"], reason))
    for k, rep in enumerate(report["gram_reports"], start=1):
        if rep["identity_ok"] and rep["psd_ok"]:
            continue
        failures.append(("gram", k, rep["size"], rep["identity_ok"],
                         rep["psd_ok"], rep["psd_reason"]))
    for label, k, size, identity_ok, psd_ok, reason in failures[:6]:
        reason = str(reason)
        if len(reason) > 180:
            reason = reason[:177] + "..."
        print(f"  fail {label}[{k}] {size}x{size}: "
              f"id={'OK' if identity_ok else 'FAIL'}, "
              f"psd={'OK' if psd_ok else 'FAIL'} ({reason})")
    if len(failures) > 6:
        print(f"  ... {len(failures) - 6} more matrix failures omitted")


# ===========================================================================
# Solve and return the exact verification report.

def solve_and_extract_report(s, l, n, m, q, d_L, **kwargs):
    """Like solve_and_extract, but also return the exact verification report."""
    holder = {}
    res = solve_sdp_v2(s=s, l=l, n=n, m=m, q=q, d_L=d_L,
                       _report_sink=holder, **kwargs)
    report = holder.get("report")
    if res.status == "certified" and report and report.get("ok"):
        return res, report.get("L_exact"), report
    return res, None, report


# ===========================================================================
# Entry point
# ===========================================================================


if __name__ == "__main__":
    from sweep_deflated import build_parser, run_sweep
    run_sweep(build_parser().parse_args())

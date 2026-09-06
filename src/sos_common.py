"""Shared game construction and SOS primitives for the submitted solvers."""
import functools
import numpy as np
import picos as pic
import sympy as sp
from SumOfSquares import SOSProblem
from SumOfSquares.basis import Basis, poly_variable
from bertrand import Bertrand

DEMAND_NAMES = {0: "all-or-nothing", 1: "linear", 2: "quadratic"}
Q_NAMES = {2: "Euclidean", 1: "entropic", 0: "Burg"}

class BertrandSOSProblem(SOSProblem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_sos_constraints = []

    def add_sos_constraint(self, expr, variables, name="", sparse=False):
        from SumOfSquares.SoS import SOSConstraint

        self._sos_const_count += 1
        name = name or f"_Q{self._sos_const_count}"
        variables_sorted = sorted(variables, key=str)
        poly = sp.poly(expr, variables_sorted)
        deg = poly.total_degree()

        half = (deg + 1) // 2   # ceil(deg/2)
        basis = Basis.from_degree(len(variables_sorted), half)

        mono_to_coeffs = dict(
            zip(poly.monoms(), map(self.sp_to_picos, poly.coeffs()))
        )

        Q = pic.SymmetricVariable(name, len(basis))
        for mono, pairs in basis.sos_sym_entries.items():
            coeff = mono_to_coeffs.get(mono, 0)
            self.add_constraint(sum(Q[i, j] for i, j in pairs) == coeff)

        pic_const = self.add_constraint(Q >> 0)
        con = SOSConstraint(pic_const, Q, basis, variables_sorted, deg)
        con.param_poly = poly
        self.all_sos_constraints.append(con)
        return con


def _total_degree(expr, x_list):
    return sp.Poly(sp.expand(expr), *x_list).total_degree()


@functools.lru_cache(maxsize=None)
def build_game(s, l, n, m):
    """(x, v, xstar, game): symbols, closed-form gradient, exact BNE.

    Cached: the symbolic gradient depends only on (s, l, n, m), so a sweep
    over q and the multiplier degrees reuses it.
    """
    game = Bertrand(s=s, l=l, n=n, m=m)
    x_int, v_int = game.get_gradient_sym()
    x = sp.symbols(f"x1:{m + 1}", positive=True)
    rename = dict(zip(x_int, x))
    v = tuple(vi.subs(rename) for vi in v_int)
    alpha = l * (n - 1) + s
    xstar = sp.Rational(alpha, alpha + 1)
    return x, v, xstar, game


@functools.lru_cache(maxsize=None)
def build_metric_matrix(s, l, n, m, metric="q", q=None):
    """Return diag(x_i**(2-q)) for the retained q=0,1,2 regularizers."""
    if metric != "q" or q not in (0, 1, 2):
        raise ValueError("Expected metric='q' and q in {0, 1, 2}.")
    x, _, _, _ = build_game(s, l, n, m)
    return sp.diag(*[xi ** (2 - q) for xi in x])


def scaled_dynamics_poly(x, v, Hinv):
    """Return (p*Hinv*v, p), where p=prod(x) clears the game-gradient denominators."""
    m = len(v)
    p = sp.prod(x)
    pv = [sp.expand(v[j] * p) for j in range(m)]
    dtilde = [sp.expand(sum(Hinv[i, j] * pv[j] for j in range(m))) for i in range(m)]
    return dtilde, p


def define_constraints(x, m, lower_bound):
    """g = [x_1 - delta, ..., x_m - delta, m - sum_k x_k]."""
    lb = sp.Rational(lower_bound)
    return [x[i] - lb for i in range(m)] + [sp.Integer(m) - sum(x)]


def build_stacked_lyapunov(x, m, d_L, xstar):
    """Build L = Phi(y)^T M Phi(y), y = x - x*,
    Phi(y) = [y, y^2, ..., y^K]  (K = d_L // 2, componentwise powers),
    M a free symmetric (K*m) x (K*m) matrix of sympy symbols.

    Every entry of Phi(y) is y_i^k with k >= 1, hence vanishes at y=0, so
    every monomial in the expansion of L = sum_{p,q} M[p,q] Phi_p Phi_q has
    degree = deg(Phi_p) + deg(Phi_q) >= 2 -- L(x*) = 0 and grad L(x*) = 0
    therefore hold identically for every choice of M, without an equality
    constraint at the equilibrium.

    At K=1 (d_L=2) this is exactly the quadratic form L(x) = y^T M y with M
    a free symmetric matrix -- constraint_positivity_and_cap's LMI
    M >= eps1*I then makes M itself (not just a Putinar certificate) supply
    positivity, i.e. this is the quadratic-restricted solver's construction
    with an explicit PSD requirement on M instead of a Putinar-only one.

    Returns (L, grad_L, M_sym, K, dim) where M_sym is the sympy symmetric
    Matrix of coefficient symbols and dim = K*m is M's size -- both needed
    by constraint_positivity_and_cap to build the matching PICOS LMI on the
    SAME underlying decision variables.
    """
    if d_L % 2 != 0:
        raise ValueError(f"d_L must be even; got d_L={d_L}")
    K = d_L // 2

    x_list = list(x)
    y = [xi - xstar for xi in x_list]
    Phi = [yi ** k for k in range(1, K + 1) for yi in y]
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
    grad_L = [sp.diff(L, xi) for xi in x_list]

    return L, grad_L, M_sym, K, dim


def putinar(prob, expr, g, x_list, d, tag, free_face=None):
    """Impose  expr - sum_j sigma_j g_j  in SOS.

    All sigma_j are SOS except the one indexed by free_face, which is left
    unrestricted; it multiplies the constraint that is active on the face
    being certified and therefore vanishes there.

    The multiplier on the constant 1 is deliberately absent: it is
    redundant, since it can always be absorbed into the SOS slack.
    """
    mult = [poly_variable(f"{tag}_{j}", x_list, d) for j in range(len(g))]
    for j, mj in enumerate(mult):
        if j != free_face:
            prob.add_sos_constraint(mj, x_list)
    con = prob.add_sos_constraint(
        sp.expand(expr - sum(mult[j] * g[j] for j in range(len(g)))), x_list)
    return mult, con


def constraint_positivity_and_cap(prob, m, dim, M_sym, epsilon1, cap_value):
    """Impose epsilon1*E <= M <= cap_value*I, with E=diag(I_m,0).

    For L=Phi(y)^T M Phi(y), the lower bound implies
    L>=epsilon1*||y||^2; the upper bound fixes the candidate scale.
    Register M symbols before constructing the polynomial constraints."""
    M_pic = pic.block([[prob.sym_to_var(M_sym[i, j]) for j in range(dim)]
                       for i in range(dim)])
    E = np.zeros((dim, dim))
    E[:m, :m] = np.eye(m)
    eps1 = float(epsilon1)
    prob.add_constraint(M_pic - eps1 * pic.Constant(E) >> 0)
    prob.add_constraint(cap_value * pic.Constant(np.eye(dim)) - M_pic >> 0)
    return M_pic


def constraint_boundary(prob, x, m, d, g, grad_L, Hinv, face, gamma):
    """Certify <grad L,Hinv grad g_face> <= gamma on the face g_face=0."""
    x_list = list(x)
    dg = [sp.diff(g[face], xi) for xi in x_list]
    ip = sum(grad_L[i] * Hinv[i, j] * dg[j] for i in range(m) for j in range(m))
    print(f"    [boundary face={face}] deg(<grad L, Hinv grad g_face>) = "
          f"{_total_degree(ip, x_list)}, multiplier degree {d}")
    return putinar(prob, -ip + gamma, g, x_list, d, f"s{face}", free_face=face)


def rationalize_scalar(val, max_den=10 ** 6):
    """Round a float to a nearby rational with bounded denominator."""
    return sp.Rational(float(val)).limit_denominator(max_den)


def sdp_size(prob):
    """Number of scalar decision variables registered in the SOS problem."""
    return sum(v.dim for v in prob.variables.values())


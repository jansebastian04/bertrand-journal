"""Floating-point SOS Lyapunov search for the Bayesian Bertrand frontier.

This SDP finds numerical Lyapunov functions without exact verification.
solver_deflated.py modifies it to enable exact rational verification,
which currently succeeds for only a small subset of instances.

The q-metric is Hinv=diag(x_i**(2-q)), q in {0,1,2}. On the truncated
simplex x_i>=delta, sum(x)<=m, use L=Phi(x-x*)^T M Phi(x-x*) with
tau*diag(I_m,0)<=M<=I. Decrease certifies dL/dt<=-c*r*L, where r is
the absolute spectral abscissa of Hinv(x*) Dv(x*). Boundary constraints
certify compatibility with each face. Shared SOS primitives live in
src/sos_common.py. This solver reports numerical feasibility; exact
rational verification is implemented in solver_deflated.py.

Use scripts/reproduce.py to replay the parameter rows stored in results.
"""

import contextlib
import csv
import io
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from sos_common import (
    DEMAND_NAMES, Q_NAMES,
    BertrandSOSProblem,
    _total_degree,
    build_game,
    build_metric_matrix,
    scaled_dynamics_poly,
    define_constraints,
    build_stacked_lyapunov,
    putinar,
    constraint_positivity_and_cap,
    constraint_boundary,
    sdp_size,
    rationalize_scalar,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


# ===========================================================================
# Precomputed rate r = |alpha(A)|, A = H(x*)^{-1} Dv(x*)
# ===========================================================================

def spectral_rate(x, v, xstar, Hinv, max_den=10 ** 6):
    """Return rationalized r=abs(alpha(A)) and float alpha(A), A=Hinv(x*) Dv(x*).
    
    The symbolic matrix is exact; eigenvalues are evaluated numerically.
    Raise RuntimeError when alpha(A)>=0 (the linearization is not Hurwitz)."""
    m = len(x)
    subs = {xi: xstar for xi in x}
    Hinv_star = Hinv.subs(subs)
    Jsym = sp.Matrix(m, m, lambda i, j: sp.diff(v[i], x[j]))
    J_star = Jsym.subs(subs)
    A = Hinv_star * J_star
    A_num = np.array(A.tolist(), dtype=float)
    alpha = float(np.max(np.linalg.eigvals(A_num).real))
    if alpha >= 0:
        raise RuntimeError(
            f"A = Hinv(x*)@Dv(x*) is not Hurwitz (alpha(A)={alpha:.6g} >= 0); "
            f"no rate r exists to normalise Decrease against.")
    r = rationalize_scalar(abs(alpha), max_den)
    return r, alpha


# ===========================================================================
# Decrease, relative to L itself: -(1/r)<grad L, wtilde> - c*p(x)*L in SOS
# (Boundary and Scale-fixing need no new function -- see
# build_sdp_problem_rescaled, which reuses constraint_boundary with its
# gamma argument fixed to 0, and constraint_positivity_and_cap with
# epsilon1 -> tau, cap_value -> 1.0, directly.)
# ===========================================================================

def constraint_decrease_rescaled(prob, x, m, d, g, L, grad_L, wtilde, prod_x, r, c):
    """Impose -(1/r)<grad L,wtilde>-c*p*L-sum(chi_j*g_j) in SOS."""
    x_list = list(x)
    if not (0 < c < 2):
        raise ValueError(f"c must lie in (0,2); got c={c}")
    cr_exact = sp.Rational(str(c)) * r
    inner = sp.expand(-sum(grad_L[i] * wtilde[i] for i in range(m)))
    pL = sp.expand(prod_x * L)
    target = sp.expand(inner - cr_exact * pL)
    print(f"    [decrease] deg(-<grad L, wtilde>) = {_total_degree(inner, x_list)}, "
          f"deg(p*L) = {_total_degree(pL, x_list)}, r={float(r):.6g}, c={c}, "
          f"c*r={float(cr_exact):.6g}, multiplier degree {d}")
    return putinar(prob, target, g, x_list, d, "chi")


def multiplier_degree_floor(intrinsic_deg):
    """Smallest nonnegative even multiplier degree at least intrinsic_degree-1."""
    if intrinsic_deg <= 0:
        return 0
    needed = intrinsic_deg - 1
    return needed if needed % 2 == 0 else needed + 1


def decrease_intrinsic_degree_rescaled(s, l, n, m, d_L, metric, q=None):
    """Intrinsic polynomial degree of the rescaled decrease expression."""
    x, v, xstar, _ = build_game(s, l, n, m)
    Hinv = build_metric_matrix(s, l, n, m, metric, q=q)
    wtilde, prod_x = scaled_dynamics_poly(x, v, Hinv)
    L, grad_L, _, _, _ = build_stacked_lyapunov(x, m, d_L, xstar)
    inner = sp.expand(-sum(grad_L[i] * wtilde[i] for i in range(m)))
    pL = sp.expand(prod_x * L)
    target = sp.expand(inner - pL)
    return _total_degree(target, list(x))


def boundary_intrinsic_degree_rescaled(s, l, n, m, d_L, metric, q=None):
    """Maximum intrinsic polynomial degree of the boundary expressions."""
    x, v, xstar, _ = build_game(s, l, n, m)
    Hinv = build_metric_matrix(s, l, n, m, metric, q=q)
    _, grad_L, _, _, _ = build_stacked_lyapunov(x, m, d_L, xstar)
    ip0 = sp.expand(sum(grad_L[i] * Hinv[i, 0] for i in range(m)))
    return _total_degree(ip0, list(x))


# ===========================================================================
# Build / solve
# ===========================================================================

def build_sdp_problem_rescaled(s, l, n, m, d_L, d_dec, d_bnd,
                               tau, c, lower_bound,
                               metric="q", q=None,
                               r_max_den=10 ** 6, verbose=False):
    """Construct the BertrandSOSProblem for the rescaled decision problem
    and return (prob, L, x, xstar, M_sym, dim, r, alpha) -- without solving.

    Raises RuntimeError (propagated from spectral_rate) if A = H(x*)^{-1}
    Dv(x*) is not Hurwitz under this metric.
    """
    x, v, xstar, _ = build_game(s, l, n, m)
    if verbose:
        print(f"  BNE x* = {xstar}")

    g = define_constraints(x, m, lower_bound)
    Hinv = build_metric_matrix(s, l, n, m, metric, q=q)
    wtilde, prod_x = scaled_dynamics_poly(x, v, Hinv)

    r, alpha = spectral_rate(x, v, xstar, Hinv, max_den=r_max_den)
    if verbose:
        print(f"  alpha(A) = {alpha:.6g}  =>  r = |alpha(A)| = {float(r):.6g} "
              f"(rationalised, A = Hinv(x*) @ Dv(x*))")

    prob = BertrandSOSProblem()

    L, grad_L, M_sym, K, dim = build_stacked_lyapunov(x, m, d_L, xstar)
    if verbose:
        print(f"    [scale-fixing] deg(L) = {_total_degree(L, list(x))}, "
              f"M is {dim}x{dim} (K={K} power layers), direct LMI (no Putinar)")
    constraint_positivity_and_cap(prob, m, dim, M_sym, tau, 1.0)
    constraint_decrease_rescaled(prob, x, m, d_dec, g, L, grad_L, wtilde, prod_x, r, c)

    # deg(<grad L, Hinv grad g_i>) is the SAME for every face i (g is
    # affine, so grad g_i is a constant vector -- see
    # boundary_intrinsic_degree_rescaled), so it is printed once here
    # rather than once per face; constraint_boundary's own per-call print
    # (from src/sos_common.py) is suppressed below to
    # avoid printing the identical line m+1 times.
    dg0 = [sp.diff(g[0], xi) for xi in list(x)]
    ip0 = sp.expand(sum(grad_L[i] * Hinv[i, j] * dg0[j]
                        for i in range(m) for j in range(m)))
    print(f"    [boundary] deg(<grad L, Hinv grad g_i>) = "
          f"{_total_degree(ip0, list(x))} (same for every face), "
          f"multiplier degree {d_bnd}")
    for face in range(m + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            constraint_boundary(prob, x, m, d_bnd, g, grad_L, Hinv, face, sp.Integer(0))

    return prob, L, x, xstar, M_sym, dim, r, alpha


def _solve_feasibility(prob, solver, timelimit=None):
    """Solve prob as a feasibility ("find") problem, returning
    (feasible, solver_used, detail).

    feasible=True iff a point satisfying every constraint was found
    (primalStatus "optimal"/"feasible" for a "find" objective -- verified
    empirically: PICOS reports "optimal" for a feasible "find" problem on
    both cvxopt and mosek, and raises SolutionFailure with a message
    containing "infeasible" when the solver confidently proves the
    opposite). feasible=False with detail containing "infeasible" is a
    genuine infeasibility proof; feasible=False otherwise (e.g. a timelimit
    cutoff before either conclusion) is inconclusive, NOT a proof of
    infeasibility -- check `detail`.
    """
    import picos
    prob.set_objective("find")
    kwargs = {} if timelimit is None else {"timelimit": timelimit}
    try:
        sol = prob.solve(solver=solver, **kwargs)
        return True, solver, getattr(sol, "primalStatus", "unknown")
    except picos.modeling.problem.SolutionFailure as exc:
        print(f"  [{solver}] {exc}")
        return False, solver, str(exc)
    except Exception as exc:
        print(f"  [{solver}] failed ({exc!r}); falling back to cvxopt.")
        try:
            sol = prob.solve(solver="cvxopt", **kwargs)
            return True, "cvxopt", getattr(sol, "primalStatus", "unknown")
        except picos.modeling.problem.SolutionFailure as exc2:
            print(f"  [cvxopt] {exc2}")
            return False, "cvxopt", str(exc2)


@dataclass
class RescaledCertificateResult:
    s: int
    l: int
    n: int
    m: int
    metric: str
    q: Optional[int]
    d_L: int
    d_dec: int
    d_bnd: int
    n_sdp: int
    tau: float
    c: float
    delta: str
    alpha: float
    r: float
    status: str
    runtime_s: float
    solver: str
    solver_status: str
    c_floor: Optional[float] = None  # Reserved empty column in the archived CSV schema.
    cond_P: Optional[float] = None  # Reserved empty column in the archived CSV schema.
    L: Optional[str] = field(default=None, repr=False)

    def row(self):
        d = asdict(self)
        d.pop("L")
        return d


CSV_FIELDS = [f for f in RescaledCertificateResult.__dataclass_fields__ if f != "L"]


def solve_sdp_rescaled(s, l, n, m, d_L, d_dec=None,
                       metric="q", q=None,
                       tau=1e-3, c=1.0,
                       lower_bound=sp.Rational(1, 20),
                       d_bnd=None,
                       delta_stab=0.0,
                       solver="mosek",
                       timelimit=900,
                       r_max_den=10 ** 6,
                       verbose=True):
    """Solve the rescaled feasibility SDP.
    
    Even d_L defines the candidate degree. None for d_dec/d_bnd selects
    the natural even multiplier floor. Certification uses numerical solver
    status, without exact rational verification. L is returned for plotting."""
    if metric == "q" and q is None:
        raise ValueError("metric='q' requires an integer q in {0,1,2}.")
    if not (0 < c < 2):
        raise ValueError(f"c must lie in (0,2); got c={c}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0; got tau={tau}")
    if d_bnd is None:
        bnd_intrinsic = boundary_intrinsic_degree_rescaled(
            s, l, n, m, d_L, metric, q=q)
        d_bnd = multiplier_degree_floor(bnd_intrinsic)
    if d_dec is None:
        intrinsic = decrease_intrinsic_degree_rescaled(
            s, l, n, m, d_L, metric, q=q)
        d_dec = multiplier_degree_floor(intrinsic)
    t0 = time.perf_counter()

    if verbose:
        print("=" * 72)
        print(f"  Game        : s={s} ({DEMAND_NAMES.get(s, s)}), l={l}, n={n}, m={m}")
        if metric == "q":
            print(f"  Regularizer : q={q} ({Q_NAMES.get(q, q)})")
        else:
            print(f"  Regularizer : {metric}")
        print(f"  Degrees     : d_L={d_L}, d_dec={d_dec}, d_bnd={d_bnd}")
        print(f"  tau={tau:g}, c={c:g}, delta={lower_bound}, delta_stab={delta_stab:g}")
        print(f"  Scale fixing: tau*E <= M <= I_(Km)")
        print(f"  Decrease    : -(1/r)<grad L,wtilde> - c*p*L - sum chi_i g_i in SOS")
        print(f"  Boundary    : -<grad L,Hinv grad g_i> - varphi_i g_i "
              f"- sum_(j!=i) eta_(i,j) g_j in SOS")
        print("=" * 72)

    try:
        prob, L, x, xstar, M_sym, dim, r, alpha = build_sdp_problem_rescaled(
            s, l, n, m, d_L, d_dec, d_bnd, tau, c, lower_bound,
            metric=metric, q=q,
            r_max_den=r_max_den, verbose=verbose)
    except RuntimeError as exc:
        if verbose:
            print(f"  ABORT: {exc}")
        return RescaledCertificateResult(
            s, l, n, m, metric, q, d_L, d_dec, d_bnd, 0, tau, c,
            str(lower_bound), float("nan"), float("nan"), "not_hurwitz",
            time.perf_counter() - t0, solver, "not_attempted", L=None)

    if float(r) < delta_stab:
        if verbose:
            print(f"  ABORT: r={float(r):.6g} < delta_stab={delta_stab:g} -- "
                  f"instance not admissible at this stability threshold.")
        return RescaledCertificateResult(
            s, l, n, m, metric, q, d_L, d_dec, d_bnd, sdp_size(prob), tau, c,
            str(lower_bound), alpha, float(r), "below_delta_stab",
            time.perf_counter() - t0, solver, "not_attempted", L=None)

    n_sdp = sdp_size(prob)
    if verbose:
        print(f"  SDP size    : {n_sdp} scalar variables")
        cap_note = f", timelimit={timelimit:g}s" if timelimit else ""
        print(f"  Solving (feasibility) with '{solver}'{cap_note} ...", flush=True)

    feasible, solver_used, detail = _solve_feasibility(prob, solver, timelimit=timelimit)

    L_val = None
    if feasible and detail in ("optimal", "feasible"):
        status = "certified"
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
            print(f"  FEASIBLE: a Lyapunov certificate at rate c*r={c * float(r):.6g} "
                  f"was found.")
        elif status == "infeasible":
            print("  INFEASIBLE: no certificate of this shape (d_L,d_dec,d_bnd,tau,c) "
                  "exists -- raise d_dec (then d_L,d_bnd), or lower c, before "
                  "concluding anything about the dynamics themselves.")
        else:
            print(f"  INCONCLUSIVE (solver status {detail!r}) -- if this hit "
                  f"timelimit={timelimit}s, it is NOT a proof of infeasibility.")

    return RescaledCertificateResult(
        s, l, n, m, metric, q, d_L, d_dec, d_bnd, n_sdp, tau, c,
        str(lower_bound), alpha, float(r), status,
        time.perf_counter() - t0, solver_used, detail, L=None if L_val is None else str(L_val))


# ===========================================================================
# Instance-list sweep with degree escalation and CSV resume.

def read_done_keys_rescaled(csv_path):
    """Read completed (s,l,n,m,metric,q) keys, including unsuccessful searches."""
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return set()
    done = set()
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            metric = row.get("metric", "q")
            q_raw = row.get("q", "")
            q_val = None if q_raw in ("", "None") else int(q_raw)
            done.add((int(row["s"]), int(row["l"]), int(row["n"]),
                      int(row["m"]), metric, q_val))
    return done


def solve_targeted_rescaled(instances, d_L_list=(2, 4, 6, 8, 10),
                            d_bnd_steps=3, d_dec_steps=3,

                            csv_path=os.path.join(RESULTS_DIR, "certificates_rescaled.csv"),
                            **kwargs):
    """Search each instance by increasing d_L, then d_bnd, then d_dec.
    
    Write one final row per instance and resume from existing CSV keys.
    Keyword arguments are forwarded to solve_sdp_rescaled."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    done = read_done_keys_rescaled(csv_path)

    results = []
    with open(csv_path, "a" if file_exists else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            wr.writeheader()
            fh.flush()
        for idx, (s, l, n, m, metric, q) in enumerate(instances, start=1):
            key = (s, l, n, m, metric, q)
            print(f"\n### [{idx}/{len(instances)}] s={s} l={l} n={n} m={m} "
                  f"metric={metric} q={q} ###", flush=True)
            if key in done:
                print("  -> already in csv, skipping", flush=True)
                continue
            found = False
            crashed = False
            res = None
            for d_L in d_L_list:
                try:
                    dec_intrinsic = decrease_intrinsic_degree_rescaled(
                        s, l, n, m, d_L, metric, q=q)
                    bnd_intrinsic = boundary_intrinsic_degree_rescaled(
                        s, l, n, m, d_L, metric, q=q)
                except Exception as exc:
                    print(f"  d_L={d_L}: could not build Hinv "
                          f"({exc}); skipping this d_L", flush=True)
                    continue
                d_dec_floor = multiplier_degree_floor(dec_intrinsic)
                d_bnd_floor = multiplier_degree_floor(bnd_intrinsic)
                print(f"  d_L={d_L}: intrinsic decrease "
                      f"degree={dec_intrinsic}, natural d_dec floor={d_dec_floor}, "
                      f"intrinsic boundary degree={bnd_intrinsic}, "
                      f"natural d_bnd floor={d_bnd_floor}", flush=True)
                for bstep in range(d_bnd_steps):
                    d_bnd = d_bnd_floor + 2 * bstep
                    for dstep in range(d_dec_steps):
                        d_dec = d_dec_floor + 2 * dstep
                        t0 = time.perf_counter()
                        try:
                            res = solve_sdp_rescaled(
                                s=s, l=l, n=n, m=m, metric=metric, q=q, d_L=d_L, d_dec=d_dec, d_bnd=d_bnd,
                                **kwargs)
                        except Exception as exc:
                            print(f"  CRASHED at d_L={d_L} "
                                  f"d_bnd={d_bnd} d_dec={d_dec}: "
                                  f"{exc.__class__.__name__}: {exc}", flush=True)
                            res = RescaledCertificateResult(
                                s, l, n, m, metric, q, d_L, d_dec, d_bnd, 0,
                                kwargs.get("tau", 1e-3), kwargs.get("c", 1.0),
                                str(kwargs.get("lower_bound", "1/20")),
                                float("nan"), float("nan"),
                                f"error_{exc.__class__.__name__}",
                                time.perf_counter() - t0,
                                kwargs.get("solver", "mosek"), "crashed", L=None)
                            crashed = True
                            break
                        if res.status in ("certified", "not_hurwitz", "below_delta_stab"):
                            found = True
                            break
                    if found or crashed:
                        break
                if found or crashed:
                    break
            wr.writerow(res.row())
            fh.flush()
            results.append(res)
            print(f"  -> FINAL: s={s} l={l} n={n} m={m} metric={metric} q={q}: "
                  f"{res.status} (r={res.r:.3g}, d_L={res.d_L}, d_dec={res.d_dec}, "
                  f"d_bnd={res.d_bnd})", flush=True)
    print(f"\nDone. {len(results)} new rows written to {csv_path}.")
    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import runpy
    runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "reproduce.py"), run_name="__main__")

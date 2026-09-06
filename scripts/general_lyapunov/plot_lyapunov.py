"""Plot rescaled Lyapunov level sets and q-metric dynamics for m=2.

Run with --from-csv to recover candidates using the archived rows' degrees,
tau, c, and delta. The CSV records solver outcomes but not the polynomial,
so plotting requires a new numerical SDP solve. Use --out-dir to keep
regenerated figures separate from the submitted PDFs.
"""

import csv
import os
import sys
import argparse
from fractions import Fraction

import numpy as np
import matplotlib
if "--no-show" in sys.argv or "--all-nine" in sys.argv or "--from-csv" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import sympy as sp
from sympy import lambdify

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(os.path.dirname(__file__))
from solver_rescaled import (
    solve_sdp_rescaled, solve_targeted_rescaled,
    build_game, build_metric_matrix, scaled_dynamics_poly,
    RESULTS_DIR, DEMAND_NAMES, Q_NAMES,
)
from bertrand import Bertrand

# ---------------------------------------------------------------------------
# Shared output directory  (same as plot.py)
# ---------------------------------------------------------------------------

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "plots")

DEFAULT_PLOTS_CSV = os.path.join(RESULTS_DIR, "certificates_rescaled_plots.csv")

# Per-instance overrides for render_lyapunov_plot's (levels, level_frac) --
# {(s, q): (levels, level_frac)}. Everything not listed here uses the
# shared --levels/(0.02, 0.65) default; add an entry only when a
# particular instance's own L is skewed enough that the default leaves
# contours invisible (see render_lyapunov_plot's docstring) -- currently
# just s=0, q=0 (Burg, all-or-nothing, m=2).
LEVEL_OVERRIDES = {
    (0, 0): (16, (0.002, 0.5)),
}


def _levels_for(s, q, default_levels):
    """(levels, level_frac) for instance (s,q), applying LEVEL_OVERRIDES
    when present."""
    return LEVEL_OVERRIDES.get((s, q), (default_levels, (0.02, 0.65)))

# ---------------------------------------------------------------------------
# Core rendering: draws ONE already-certified RescaledCertificateResult
# (streamplot of the gradient field + Lyapunov level-set contours over
# B_delta^m). Solving and rendering are deliberately separate: --all-nine
# needs to solve every instance FIRST (via solve_targeted_rescaled's
# resumable escalation search), then render only the ones that came back
# certified, reusing this exact same rendering logic as the single-
# instance CLI path.
# ---------------------------------------------------------------------------

def render_lyapunov_plot(s, l, n, m, metric, q, result, *,
                         delta,
                         grid=250, levels=8, level_frac=(0.02, 0.65),
                         density=2.0,
                         save=False, out_dir=OUT_DIR):
    """Render a certified RescaledCertificateResult as a Lyapunov level-set
    + gradient-flow streamplot over B_delta^m (m=2 only). Returns the
    matplotlib Figure; caller decides whether to show/save/close it.

    level_frac = (lo, hi): contour levels are L_min + linspace(lo, hi,
    levels) * (L_max - L_min). L's growth away from the BNE can be
    extremely skewed (observed: q=0 (Burg), s=0, m=2 has L ranging over
    ~7 orders of magnitude, with the bulk of the domain sitting under the
    5th percentile of L's range), so the default (0.02, 0.65) can leave
    almost every contour bunched up far from the BNE with nothing visible
    in between -- lower `lo` (and/or raise `levels`) for such an instance
    to pull contours back into the region that's actually informative.
    """
    if m != 2:
        raise ValueError(
            f"2-D visualisation is only possible for m = 2 (got m = {m}).\n"
            "For m > 2 the state space is higher-dimensional and cannot be "
            "plotted directly.  Consider fixing all but two coordinates and "
            "plotting a 2-D slice instead."
        )
    if result.status != "certified" or result.L is None:
        raise ValueError(
            f"result is not a certified certificate (status={result.status}).")

    lb = delta
    LB_sym = sp.Rational(delta).limit_denominator(1000)
    L_expr = sp.sympify(result.L)

    # ── Symbolic dynamics  x_dot_i = H^{-1}_{ii} * v_i(x) ──────────────────
    x_syms, v_exprs, _, _ = build_game(s, l, n, m)
    Hinv = build_metric_matrix(s, l, n, m, metric, q=q)
    dtilde_polys, prod_x_sym = scaled_dynamics_poly(x_syms, v_exprs, Hinv)
    vel_exprs = [sp.simplify(dt / prod_x_sym) for dt in dtilde_polys]

    L_fn = lambdify(x_syms, L_expr, modules="numpy")
    vel_fns = [lambdify(x_syms, ve, modules="numpy") for ve in vel_exprs]

    # ── Feasible-set geometry  B_delta^m ───────────────────────────────────
    tri_verts = np.array([
        [lb,     lb    ],
        [m - lb, lb    ],
        [lb,     m - lb],
    ])

    EPS = 1e-6
    def in_simplex(x1, x2):
        return (x1 >= lb - EPS) & (x2 >= lb - EPS) & (x1 + x2 <= m + EPS)

    # ── Evaluate L and velocity on the grid ────────────────────────────────
    xs = np.linspace(lb, m - lb, grid)
    X1, X2 = np.meshgrid(xs, xs)
    MASK = in_simplex(X1, X2)

    L_grid = np.where(MASK, L_fn(X1, X2), np.nan)
    L_grid = np.where(MASK & (L_grid < 0), 0.0, L_grid)

    with np.errstate(invalid="ignore", divide="ignore"):
        V1_raw = np.where(MASK, vel_fns[0](X1, X2), 0.0)
        V2_raw = np.where(MASK, vel_fns[1](X1, X2), 0.0)

    v_cap = np.nanpercentile(np.abs(np.stack([V1_raw, V2_raw])), 99)
    V1 = np.clip(V1_raw, -v_cap, v_cap)
    V2 = np.clip(V2_raw, -v_cap, v_cap)

    # ── BNE ─────────────────────────────────────────────────────────────────
    bne_val = Bertrand(s=s, l=l, n=n, m=m).bne[0]
    xstar = float(Fraction(bne_val).limit_denominator(1000))

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    stream = ax.streamplot(
        xs, xs, V1, V2,
        color="black",
        linewidth=0.7,
        density=density,
        arrowsize=0.9,
        arrowstyle="-|>",
        zorder=2,
    )

    tri_path_verts = [
        (tri_verts[0, 0], tri_verts[0, 1]),
        (tri_verts[1, 0], tri_verts[1, 1]),
        (tri_verts[2, 0], tri_verts[2, 1]),
        (tri_verts[0, 0], tri_verts[0, 1]),
    ]
    tri_codes  = [mpath.Path.MOVETO, mpath.Path.LINETO,
                  mpath.Path.LINETO, mpath.Path.CLOSEPOLY]
    tri_path   = mpath.Path(tri_path_verts, tri_codes)
    clip_patch = mpatches.PathPatch(tri_path, transform=ax.transData,
                                    facecolor="none")
    ax.add_patch(clip_patch)
    stream.lines.set_clip_path(clip_patch)
    stream.arrows.set_clip_path(clip_patch)

    L_min = np.nanmin(L_grid)
    L_max = np.nanmax(L_grid)
    lo_frac, hi_frac = level_frac
    lvls = L_min + np.linspace(lo_frac, hi_frac, levels) * (L_max - L_min)
    ax.contour(X1, X2, L_grid,
               levels=lvls, colors="steelblue",
               linewidths=1.1, alpha=0.85, zorder=3)

    tri_xs = [tri_verts[0, 0], tri_verts[1, 0], tri_verts[2, 0], tri_verts[0, 0]]
    tri_ys = [tri_verts[0, 1], tri_verts[1, 1], tri_verts[2, 1], tri_verts[0, 1]]
    ax.plot(tri_xs, tri_ys, color="black", linewidth=2.5,
            zorder=4, solid_capstyle="round")

    ax.scatter([xstar], [xstar], color="red", s=80, zorder=5)

    legend_handles = [
        mpatches.Patch(facecolor="none", edgecolor="steelblue",
                       linewidth=1.5, label="Lyapunov Level Sets"),
        mpatches.Patch(facecolor="none", edgecolor="black",
                       linewidth=2.5, label=r"Feasible Region $\mathcal{B}_\delta^m$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=8, label="BNE"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              framealpha=0.9, edgecolor="gray")

    delta_str = str(LB_sym)
    reg_title = f"{Q_NAMES[q]} Dynamics  ($q={q}$)"
    ax.set_title(
        f"{reg_title}  —  {DEMAND_NAMES[s]} model  ($s={s}$)\n"
        f"$\\ell={l}$,  $n={n}$,  $m={m}$,  $\\delta={delta_str}$,  "
        f"$d_L={result.d_L}$,  $d_{{dec}}={result.d_dec}$,  $d_{{bnd}}={result.d_bnd}$\n"
        f"$\\tau={result.tau:g}$,  $r={result.r:.3g}$",
        fontsize=12,
    )

    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)
    ax.set_xlim(-0.02, m + 0.05)
    ax.set_ylim(-0.02, m + 0.05)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", color="gray", alpha=0.4, zorder=0)
    ax.tick_params(labelsize=11)

    plt.tight_layout()

    if save:
        os.makedirs(out_dir, exist_ok=True)
        reg_tag = f"q{q}" if metric == "q" else metric
        fname = (f"lyapunov_rescaled_{reg_tag}_m{m}_s{s}_l{l}_n{n}_"
                f"dL{result.d_L}_ddec{result.d_dec}_dbnd{result.d_bnd}.pdf")
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved -> {os.path.abspath(out_path)}")

    return fig


# ---------------------------------------------------------------------------
# Single-instance mode: solve exactly the requested (d_L, d_dec, d_bnd) --
# no escalation, matching this file's old direct/one-shot workflow. Pass
# --d-dec/--d-bnd as None (the default) to fall back to solve_sdp_rescaled's
# own natural-floor degree, without searching further if that floor turns
# out infeasible.
# ---------------------------------------------------------------------------

def plot_lyapunov(args):
    if args.m != 2:
        raise ValueError(
            f"2-D visualisation is only possible for m = 2 (got m = {args.m})."
        )

    lb_sym = sp.Rational(args.delta).limit_denominator(1000)

    reg_desc = f"q={args.q}"
    print(f"Solving rescaled SDP  (s={args.s}, l={args.l}, n={args.n}, m={args.m}, "
          f"metric={args.metric} ({reg_desc}), dL={args.dL}, d_dec={args.d_dec}, "
          f"d_bnd={args.d_bnd}, tau={args.tau}, c={args.c}, delta={args.delta}) ...")
    result = solve_sdp_rescaled(
        s=args.s, l=args.l, n=args.n, m=args.m,
        metric=args.metric, q=args.q,
        d_L=args.dL, d_dec=args.d_dec, d_bnd=args.d_bnd,
        tau=args.tau, c=args.c, lower_bound=lb_sym, delta_stab=args.delta_stab,
        solver=args.solver, timelimit=args.timelimit,
    )

    if result.status != "certified" or result.L is None:
        raise RuntimeError(
            f"SDP returned no certificate (status = {result.status}).  Try a "
            "larger --dL/--d-dec/--d-bnd, a different --tau/--c, or use "
            "--all-nine's automatic degree escalation."
        )
    print(f"Certificate found  (d_L={result.d_L}, d_dec={result.d_dec}, "
          f"d_bnd={result.d_bnd}, r={result.r:.3g}, c*r={result.c * result.r:.3g})")

    lvls, lvl_frac = _levels_for(args.s, args.q, args.levels)
    fig = render_lyapunov_plot(
        args.s, args.l, args.n, args.m, args.metric, args.q, result,
        delta=args.delta,
        grid=args.grid, levels=lvls, level_frac=lvl_frac, density=args.density,
        save=args.save, out_dir=OUT_DIR,
    )
    if not args.save:
        print("Not saving (pass --save to write a PDF to results/plots/).")
    return fig


# ---------------------------------------------------------------------------
# --all-nine: s in {0,1,2}, q in {0,1,2}, metric="q", l/n/m fixed (default
# l=1, n=2, m=2 -- exactly the nine requested instances). Degrees are found
# via solve_targeted_rescaled's resumable escalation search (cached in
# --csv). solve_targeted_rescaled's own return value already carries a
# full RescaledCertificateResult -- L included -- for every instance it
# actually solved THIS call, so those are rendered directly with NO
# further solving. Only an instance skipped as "already in csv" (cached
# from an EARLIER, separate run) has no in-memory L (RescaledCertificate
# Result.row() deliberately drops L before writing -- see that dataclass),
# so only THOSE are re-solved at their cached degrees, with a few retries:
# this SDP sits at (tau, c) values close enough to the feasibility
# boundary that a fresh MOSEK solve at the EXACT SAME degrees occasionally
# comes back "unknown" instead of "optimal" (observed directly: the first
# --all-nine run here certified all nine during the search, but re-solving
# 4 of them afterwards to recover L failed on the first attempt) --
# a doomed instance and a numerically-borderline one look identical from a
# single "unknown" verdict, so a retry (not a larger d_dec, which would
# silently stop testing the DEGREES the CSV says are certified) is the
# right response here. One failure does not stop the batch: it is
# reported and the run moves on.
# ---------------------------------------------------------------------------

def _load_csv_rows(csv_path, instances):
    """{(s,l,n,m,metric,q): row} for whichever of `instances` already have
    a row in csv_path (freshly solved this call, or cached from a
    previous run -- solve_targeted_rescaled's own return value only
    covers the former)."""
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return {}
    wanted = set(instances)
    out = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            metric = row.get("metric", "q")
            q_raw = row.get("q", "")
            q_val = None if q_raw in ("", "None") else int(q_raw)
            key = (int(row["s"]), int(row["l"]), int(row["n"]),
                  int(row["m"]), metric, q_val)
            if key in wanted:
                out[key] = row
    return out


def _resolve_for_L(s, l, n, m, metric, q, d_L, d_dec, d_bnd,
                   tau, c, lower_bound, delta_stab, solver, timelimit,
                   max_attempts=3):
    """Re-solve at exactly the given (already-certified) degrees/constants
    to recover L, retrying a few times before giving up -- see
    plot_all_nine's docstring comment on why a retry, not an escalation,
    is the right response to a single "unknown" verdict here."""
    for attempt in range(1, max_attempts + 1):
        result = solve_sdp_rescaled(
            s=s, l=l, n=n, m=m, metric=metric, q=q,
            d_L=d_L, d_dec=d_dec, d_bnd=d_bnd,
            tau=tau, c=c, lower_bound=lower_bound, delta_stab=delta_stab,
            solver=solver, timelimit=timelimit, verbose=False,
        )
        if result.status == "certified" and result.L is not None:
            return result
        print(f"    attempt {attempt}/{max_attempts}: status={result.status}, retrying ..."
              if attempt < max_attempts else
              f"    attempt {attempt}/{max_attempts}: status={result.status}, giving up.")
    return result


def _s_q_list(args):
    """(s_list, q_list) from --s-list/--q-list, default (0,1,2) each --
    lets a re-plot target a subset (e.g. just q=2) without touching the
    other, already-good instances."""
    s_list = tuple(int(v) for v in args.s_list.split(","))
    q_list = tuple(int(v) for v in args.q_list.split(","))
    return s_list, q_list


def plot_all_nine(args):
    lb_sym = sp.Rational(args.delta).limit_denominator(1000)
    d_L_list = tuple(int(v) for v in args.d_L_list.split(","))
    s_list, q_list = _s_q_list(args)
    instances = [(s, args.l, args.n, args.m, "q", q)
                for s in s_list for q in q_list]

    print(f"Searching certificates for {len(instances)} (s,q) instances "
          f"(l={args.l}, n={args.n}, m={args.m}, tau={args.tau}, c={args.c}, "
          f"delta={args.delta}) ...")
    fresh_results = solve_targeted_rescaled(
        instances,
        d_L_list=d_L_list, d_bnd_steps=args.d_bnd_steps, d_dec_steps=args.d_dec_steps,
        csv_path=args.csv,
        lower_bound=lb_sym, tau=args.tau, c=args.c, delta_stab=args.delta_stab,
        solver=args.solver, timelimit=args.timelimit,
    )
    fresh_by_key = {(r.s, r.l, r.n, r.m, r.metric, r.q): r for r in fresh_results}
    cached_rows = _load_csv_rows(args.csv, instances)

    ok, skipped = 0, []
    for (s, l, n, m, metric, q) in instances:
        key = (s, l, n, m, metric, q)
        result = fresh_by_key.get(key)
        if result is not None:
            if result.status != "certified" or result.L is None:
                print(f"  [skip] s={s} q={q}: status={result.status}, not plotting.")
                skipped.append((s, q, result.status))
                continue
        else:
            # Not solved this call -> cached from an earlier, separate run;
            # the CSV has no L, so recover it with a retrying re-solve.
            row = cached_rows.get(key)
            if row is None or row["status"] != "certified":
                status = row["status"] if row else "missing"
                print(f"  [skip] s={s} q={q}: status={status}, not plotting.")
                skipped.append((s, q, status))
                continue
            print(f"  [plot] s={s} q={q}: cached from a previous run -- "
                  f"re-solving at its certified degrees (d_L={row['d_L']}, "
                  f"d_dec={row['d_dec']}, d_bnd={row['d_bnd']}) to recover L "
                  f"(not stored in the CSV) ...")
            result = _resolve_for_L(
                s, l, n, m, metric, q,
                int(row["d_L"]), int(row["d_dec"]), int(row["d_bnd"]),
                args.tau, args.c, lb_sym, args.delta_stab, args.solver, args.timelimit)
            if result.status != "certified" or result.L is None:
                print(f"  [plot-failed] s={s} q={q}: re-solve did not reproduce "
                      f"a certificate after retries (status={result.status}); "
                      f"skipping plot ({args.csv} still records it as certified).")
                skipped.append((s, q, f"resolve_{result.status}"))
                continue
        try:
            lvls, lvl_frac = _levels_for(s, q, args.levels)
            fig = render_lyapunov_plot(
                s, l, n, m, "q", q, result,
                delta=args.delta, grid=args.grid, levels=lvls, level_frac=lvl_frac,
                density=args.density, save=True, out_dir=OUT_DIR)
            plt.close(fig)
            ok += 1
        except Exception as exc:
            print(f"  [plot-failed] s={s} q={q}: {exc.__class__.__name__}: {exc}")
            skipped.append((s, q, "plot_error"))

    print(f"\nDone: {ok}/{len(instances)} plotted and saved to {OUT_DIR}.")
    if skipped:
        print("Not plotted:")
        for s, q, why in skipped:
            print(f"  s={s} q={q}: {why}")


# ---------------------------------------------------------------------------
# --from-csv: plot the nine (s,q) instances at --l/--n/--m directly from an
# EXISTING, already-populated certificates CSV -- e.g. results/certificates_
# rescaled.csv, written by solver_rescaled.py's own
# __main__ sweep -- with NO fresh search. Each row's own (d_L, d_dec,
# d_bnd, tau, c, delta) is used exactly as recorded (NOT --dL/--tau/--c/
# --delta, which are irrelevant in this mode and may not even match --
# a sweep run separately can certify at different, often LOWER, degrees
# for the same instance under a different tau). Only L is missing from
# the CSV and is recovered the same way as plot_all_nine's cached-row
# fallback: a retrying re-solve at those exact recorded values.
# ---------------------------------------------------------------------------

def plot_from_csv(args):
    s_list, q_list = _s_q_list(args)
    instances = [(s, args.l, args.n, args.m, "q", q)
                for s in s_list for q in q_list]
    print(f"Plotting {len(instances)} (s,q) instances (l={args.l}, n={args.n}, m={args.m}) "
          f"from {args.from_csv} -- each row's own recorded degrees/tau/c/delta ...")
    rows = _load_csv_rows(args.from_csv, instances)

    ok, skipped = 0, []
    for (s, l, n, m, metric, q) in instances:
        row = rows.get((s, l, n, m, metric, q))
        if row is None or row["status"] != "certified":
            status = row["status"] if row else "missing"
            print(f"  [skip] s={s} q={q}: status={status}, not plotting.")
            skipped.append((s, q, status))
            continue
        row_tau = float(row["tau"])
        row_c = float(row["c"])
        row_delta = sp.Rational(row["delta"])
        print(f"  [plot] s={s} q={q}: re-solving at its recorded degrees "
              f"(d_L={row['d_L']}, d_dec={row['d_dec']}, d_bnd={row['d_bnd']}, "
              f"tau={row_tau:g}, c={row_c:g}, delta={row_delta}) to recover L ...")
        result = _resolve_for_L(
            s, l, n, m, metric, q,
            int(row["d_L"]), int(row["d_dec"]), int(row["d_bnd"]),
            row_tau, row_c, row_delta, args.delta_stab, args.solver, args.timelimit)
        if result.status != "certified" or result.L is None:
            print(f"  [plot-failed] s={s} q={q}: re-solve did not reproduce "
                  f"a certificate after retries (status={result.status}); "
                  f"skipping plot ({args.from_csv} still records it as certified).")
            skipped.append((s, q, f"resolve_{result.status}"))
            continue
        try:
            lvls, lvl_frac = _levels_for(s, q, args.levels)
            fig = render_lyapunov_plot(
                s, l, n, m, "q", q, result,
                delta=float(row_delta), grid=args.grid, levels=lvls, level_frac=lvl_frac,
                density=args.density, save=True, out_dir=OUT_DIR)
            plt.close(fig)
            ok += 1
        except Exception as exc:
            print(f"  [plot-failed] s={s} q={q}: {exc.__class__.__name__}: {exc}")
            skipped.append((s, q, "plot_error"))

    print(f"\nDone: {ok}/{len(instances)} plotted and saved to {OUT_DIR}.")
    if skipped:
        print("Not plotted:")
        for s, q, why in skipped:
            print(f"  s={s} q={q}: {why}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="plot_lyapunov.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--s",       type=int,   default=0,    help="demand exponent (default: 0)")
    parser.add_argument("--l",       type=int,   default=1,    help="Beta(1,l) prior parameter (default: 1)")
    parser.add_argument("--n",       type=int,   default=2,    help="number of players (default: 2)")
    parser.add_argument("--m",       type=int,   default=2,    help="strategy pieces, must be 2 (default: 2)")
    parser.add_argument("--metric",  type=str,   default="q",  choices=["q"], help="regularizer family (default: q)")
    parser.add_argument("--q",       type=int,   default=2,    help="regularizer 2/1/0, used when --metric q (default: 2)")
    parser.add_argument("--dL",      type=int,   default=4,    help="Lyapunov degree, even (default: 4)")
    parser.add_argument("--d-dec",   type=int,   default=None, dest="d_dec", help="decrease-constraint SOS multiplier degree (default: auto natural floor)")
    parser.add_argument("--d-bnd",   type=int,   default=None, dest="d_bnd", help="boundary-constraint SOS multiplier degree (default: auto natural floor)")
    parser.add_argument("--tau",     type=float, default=1e-3, help="scale-fixing constant: tau*E <= M <= I (default: 1e-3)")
    parser.add_argument("--c",       type=float, default=1.0,  help="decrease constant in (0,2): dL/dt <= -c*r*L (default: 1.0)")
    parser.add_argument("--delta",   type=float, default=1 / 250, help="lower bound on x_i (default: 1/250)")
    parser.add_argument("--delta-stab", type=float, default=0.0, dest="delta_stab", help="admissibility floor on r=|alpha(A)| (default: 0.0)")
    parser.add_argument("--grid",    type=int,   default=250,  help="grid resolution (default: 250)")
    parser.add_argument("--levels",  type=int,   default=8,    help="number of level-set curves (default: 8)")
    parser.add_argument("--density", type=float, default=2.0,  help="streamplot density (default: 2.0)")
    parser.add_argument("--solver",  type=str,   default="mosek", help="SDP solver: mosek or cvxopt (default: mosek)")
    parser.add_argument("--timelimit", type=float, default=900, help="solver timelimit in seconds (default: 900)")
    parser.add_argument("--save",    action="store_true",       help="write the figure to results/plots/*.pdf (off by default)")
    parser.add_argument("--no-show", action="store_true",       help="suppress plt.show()")

    parser.add_argument("--all-nine", action="store_true", dest="all_nine",
                        help="solve+plot all nine (s,q) in {0,1,2}x{0,1,2} instances "
                             "at --l/--n/--m (default l=1,n=2,m=2); ignores "
                             "--metric/--dL/--d-dec/--d-bnd")
    parser.add_argument("--s-list", type=str, default="0,1,2", dest="s_list",
                        help="--all-nine/--from-csv only: comma-separated s values to "
                             "plot, e.g. '0' to target a single s (default: 0,1,2)")
    parser.add_argument("--q-list", type=str, default="0,1,2", dest="q_list",
                        help="--all-nine/--from-csv only: comma-separated q values to "
                             "plot, e.g. '2' to re-plot just q=2 (default: 0,1,2)")
    parser.add_argument("--d-L-list", type=str, default="2,4,6,8,10", dest="d_L_list",
                        help="--all-nine only: comma-separated d_L escalation schedule "
                             "(default: 2,4,6,8,10)")
    parser.add_argument("--d-dec-steps", type=int, default=3, dest="d_dec_steps",
                        help="--all-nine only: d_dec escalation steps per d_L (default: 3)")
    parser.add_argument("--d-bnd-steps", type=int, default=3, dest="d_bnd_steps",
                        help="--all-nine only: d_bnd escalation steps per d_L (default: 3)")
    parser.add_argument("--csv", type=str, default=DEFAULT_PLOTS_CSV,
                        help=f"--all-nine only: resumable cache of certified degrees "
                             f"(default: {DEFAULT_PLOTS_CSV})")

    parser.add_argument("--from-csv", type=str, nargs="?",
                        const=os.path.join(RESULTS_DIR, "certificates_rescaled.csv"),
                        default=None, dest="from_csv",
                        help="plot the nine (s,q) instances at --l/--n/--m directly "
                             "from an EXISTING certificates CSV, with NO fresh search "
                             "-- each row's own recorded d_L/d_dec/d_bnd/tau/c/delta is "
                             "used exactly as-is (--dL/--tau/--c/--delta are ignored in "
                             "this mode). Bare flag defaults to "
                             "results/certificates_rescaled.csv (the archived "
                             "floating-point frontier results).")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Directory for generated PDFs")
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    global OUT_DIR
    OUT_DIR = args.out_dir
    if args.from_csv:
        plot_from_csv(args)
    elif args.all_nine:
        plot_all_nine(args)
    else:
        plot_lyapunov(args)
        if not args.no_show:
            plt.show()


if __name__ == "__main__":
    main()

"""
Sweep solver_deflated.py on the requested
q=2 grid and write the exact verified Lyapunov polynomial into the CSV.

Grid:
    s in {0,1,2}, l in {1,2,3}, n in {2,3,4}, with q=2.
    The default is m=2, matching the retained results.

Constants:
    delta = 1/250, kappa = c = 1, mu = eps_M = 100 / max_den.
    By default max_den is tried at 1e8 first, then 1e10.

The search escalates in the same spirit as the rescaled targeted sweeps:
for every d_L, solve_sdp_v2 computes the natural d_dec/d_bnd floors and this
script tries each requested bump.  The default order is d_dec bump first,
then d_bnd bump, then max_den=1e10, and only then the next d_L in {2,4,6}.
"""

import argparse
import ast
import csv
import math
import os
import time

import sympy as sp

from solver_deflated import (
    C_RATE,
    CSV_FIELDS,
    DELTA,
    KAPPA,
    MarginCertificateResultV2,
    RATE_FORMS,
    RESULTS_DIR,
    solve_and_extract_report,
)


Q_FIXED = 2
DEFAULT_M_VALUES = (2,)

DEFAULT_MAX_DEN_VALUES = (10 ** 8, 10 ** 10)
DEFAULT_Q_MAX_DEN_FACTOR = 10 ** 4
DEFAULT_D_L_LIST = (2, 4, 6)
DEFAULT_D_BND_BUMPS = (0, 2, 4)
DEFAULT_D_DEC_BUMPS = (0, 2, 4)
DEFAULT_CSV = os.path.join(
    RESULTS_DIR, "deflated_sos_basis_v2_m2_q2_verified.csv")

BASE_FIELDS = [
    field for field in CSV_FIELDS
    if field not in ("boundary_mode", "r")
]
EXTRA_FIELDS = [
    "decrease_rate",
    "max_den",
    "q_max_den",
    "mosek_tol",
    "attempt_count",
    "max_d_L_tested",
    "exhausted_to_max_d_L",
    "sweep_note",
    "min_projected_lambda_float",
    "max_gram_basis_size",
    "rough_min_max_den_for_screen",
    "verified",
    "lyapunov_function",
]
SWEEP_FIELDS = BASE_FIELDS + EXTRA_FIELDS


def parse_int_list(text):
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def q_max_den_for(max_den, explicit_q_max_den):
    return explicit_q_max_den if explicit_q_max_den is not None else (
        max_den * DEFAULT_Q_MAX_DEN_FACTOR)


def scalar_d_bnd(value):
    if isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        text = str(value)
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
        vals = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    if not vals:
        return ""
    try:
        vals = [int(v) for v in vals]
    except (TypeError, ValueError):
        return str(value)
    return vals[0] if len(set(vals)) == 1 else max(vals)


def mosek_options(tol):
    """PICOS options that tighten only MOSEK's conic interior-point tolerances.

    Do not use PICOS' abs_prim_fsb_tol/abs_dual_fsb_tol here: PICOS maps
    those to MOSEK simplex basis tolerances too, and BASIS_TOL_X rejects
    values such as 1e-10 even though the conic IPM tolerances accept them.
    """
    if tol is None or tol <= 0:
        return None
    return {
        "mosek_params": {
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
        },
    }


def read_done_keys(csv_path):
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return set()
    done = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            decrease_rate = row.get("decrease_rate") or "pL"
            done.add((int(row["s"]), int(row["l"]), int(row["n"]),
                      int(row.get("m", 2)), int(row.get("q", Q_FIXED)),
                      decrease_rate))
    return done


def ensure_csv_schema(csv_path):
    """Add newly introduced columns to an existing sweep CSV if needed."""
    if not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0):
        return
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames == SWEEP_FIELDS:
            return
        rows = list(reader)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for row in rows:
            row = {field: row.get(field, "") for field in SWEEP_FIELDS}
            row["d_bnd"] = scalar_d_bnd(row["d_bnd"])
            if not row["decrease_rate"]:
                row["decrease_rate"] = "pL"
            writer.writerow(row)


def report_denominator_diagnostics(report, screen_tol):
    if not report:
        return "", "", ""
    entries = []
    for rep in report.get("mult_reports", []):
        info = rep.get("recovery", {})
        lam = info.get("lambda_min_float")
        if lam is not None:
            entries.append((float(lam), int(rep["size"])))
    for rep in report.get("gram_reports", []):
        info = rep.get("recovery", {})
        lam = info.get("lambda_min_float")
        if lam is not None:
            entries.append((float(lam), int(rep["size"])))
    if not entries:
        return "", "", ""
    min_lam = min(lam for lam, _ in entries)
    max_size = max(size for _, size in entries)
    if min_lam <= screen_tol:
        rough = ""
    else:
        rough = math.ceil(max_size / (2.0 * (min_lam - screen_tol)))
    return min_lam, max_size, rough


def result_row(res, L_exact, decrease_rate, max_den, q_max_den, mosek_tol,
               attempt_count, max_d_L_tested, exhausted_to_max_d_L,
               min_projected_lambda_float="", max_gram_basis_size="",
               rough_min_max_den_for_screen=""):
    verified = bool(L_exact is not None and res.status == "certified")
    if verified:
        sweep_note = f"verified at d_L={res.d_L}"
    elif exhausted_to_max_d_L:
        sweep_note = f"no verified certificate found up to d_L={max_d_L_tested}"
    else:
        sweep_note = f"stopped before exhausting d_L={max_d_L_tested}"
    raw = res.row()
    raw["d_bnd"] = scalar_d_bnd(raw["d_bnd"])
    row = {field: raw.get(field, "") for field in BASE_FIELDS}
    row.update({
        "decrease_rate": decrease_rate,
        "max_den": max_den,
        "q_max_den": q_max_den,
        "mosek_tol": "" if mosek_tol is None else mosek_tol,
        "attempt_count": attempt_count,
        "max_d_L_tested": max_d_L_tested,
        "exhausted_to_max_d_L": exhausted_to_max_d_L,
        "sweep_note": sweep_note,
        "min_projected_lambda_float": min_projected_lambda_float,
        "max_gram_basis_size": max_gram_basis_size,
        "rough_min_max_den_for_screen": rough_min_max_den_for_screen,
        "verified": verified,
        "lyapunov_function": "" if L_exact is None else str(sp.expand(L_exact)),
    })
    return row


def error_result(s, l, n, m, d_L, d_dec_bump, d_bnd_bump, exc, runtime_s,
                 eps_M, solver):
    return MarginCertificateResultV2(
        s=s,
        l=l,
        n=n,
        m=m,
        q=Q_FIXED,
        d_L=d_L,
        d_dec=-1,
        d_bnd=-1,
        n_sdp=0,
        boundary_mode="face",
        c=C_RATE,
        delta=str(DELTA),
        eps_R=None,
        eps_M=float(eps_M),
        rho=float("nan"),
        alpha=float("nan"),
        r=float("nan"),
        mu=float(eps_M),
        w1=float("nan"),
        mu_z2=float("nan"),
        status=f"error_{exc.__class__.__name__}",
        runtime_s=runtime_s,
        solver=solver,
        solver_status=(
            f"crashed at d_L={d_L}, d_dec_bump={d_dec_bump}, "
            f"d_bnd_bump={d_bnd_bump}: {exc}"),
        L=None,
    )


def better_result(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return candidate
    if candidate.status == "certified" and current.status != "certified":
        return candidate
    if candidate.eps_R is not None and current.eps_R is not None:
        return candidate if candidate.eps_R > current.eps_R else current
    if candidate.eps_R is None and current.eps_R is not None:
        return current
    return candidate


def escalation_note(d_L_i, bnd_i, dec_i, d_L_values, bnd_bumps, dec_bumps,
                    max_den_i, max_den_values, verified):
    if verified:
        return ""
    if dec_i + 1 < len(dec_bumps):
        return "increase d_dec"
    if bnd_i + 1 < len(bnd_bumps):
        return "increase d_bnd"
    if max_den_i + 1 < len(max_den_values):
        return f"increase max_den to {max_den_values[max_den_i + 1]}"
    if d_L_i + 1 < len(d_L_values):
        return "increase d_L"
    return f"instance failed through d_L={max(d_L_values)}"


def attempt_message(res, L_exact, d_dec_bump, d_bnd_bump, next_step):
    d_bnd = scalar_d_bnd(res.d_bnd)
    where = (f"d_L={res.d_L}, d_bnd={d_bnd}, d_dec={res.d_dec} "
             f"(bumps bnd={d_bnd_bump}, dec={d_dec_bump})")
    if L_exact is not None and res.status == "certified":
        return f"  VERIFIED at {where} ({res.runtime_s:.1f}s)"
    if res.status == "certified_verify_skipped":
        return (f"  certification possible at {where}, but exact verification "
                f"was skipped -> {next_step}")
    if res.status.startswith("certified"):
        return (f"  certification possible at {where}, but exact verification "
                f"did not pass -> {next_step}")
    if res.status == "infeasible":
        return f"  infeasible at {where} -> {next_step}"
    return f"  no usable certificate at {where} ({res.status}) -> {next_step}"


def run_sweep(args):
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    ensure_csv_schema(args.csv)
    done = read_done_keys(args.csv) if args.resume else set()
    new_file = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0

    base_instances = [(m, s, l, n)
                      for m in args.m_values
                      for s in args.s_values
                      for l in args.l_values
                      for n in args.n_values]
    max_den_values = tuple(args.max_den)
    d_L_values = tuple(args.d_L_list)
    d_bnd_bumps = tuple(args.d_bnd_bumps)
    d_dec_bumps = tuple(args.d_dec_bumps)
    solve_options = mosek_options(args.mosek_tol)

    with open(args.csv, "a" if not new_file else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SWEEP_FIELDS)
        if new_file:
            writer.writeheader()
            fh.flush()

        for idx, (m, s, l, n) in enumerate(base_instances, start=1):
            key = (s, l, n, m, Q_FIXED, args.decrease_rate)
            if key in done:
                print(f"\n### [{idx}/{len(base_instances)}] s={s} l={l} "
                      f"n={n} m={m} q={Q_FIXED} rate={args.decrease_rate}: "
                      f"already in csv, skipping ###", flush=True)
                continue

            print(f"\n### [{idx}/{len(base_instances)}] s={s} l={l} n={n} "
                  f"m={m} q={Q_FIXED} rate={args.decrease_rate} ###", flush=True)

            best = None
            best_L = None
            best_report = None
            best_max_den = ""
            best_q_max_den = ""
            attempts = 0
            max_d_L_tested = ""
            stop = False

            for d_L_i, d_L in enumerate(d_L_values):
                max_d_L_tested = d_L
                for max_den_i, max_den in enumerate(max_den_values):
                    q_max_den = q_max_den_for(max_den, args.q_max_den)
                    eps_M = 100.0 / max_den
                    print(f"  trying max_den={max_den}, q_max_den={q_max_den}, "
                          f"mu=eps_M={eps_M:g}", flush=True)
                    for bnd_i, d_bnd_bump in enumerate(d_bnd_bumps):
                        for dec_i, d_dec_bump in enumerate(d_dec_bumps):
                            attempts += 1
                            t0 = time.perf_counter()
                            try:
                                res, L_exact, report = solve_and_extract_report(
                                    s=s, l=l, n=n, m=m, q=Q_FIXED,
                                    d_L=d_L,
                                    d_dec=None, d_bnd=None,
                                    d_dec_bump=d_dec_bump,
                                    d_bnd_bump=d_bnd_bump,
                                    c=C_RATE,
                                    kappa=KAPPA,
                                    mult_cap=KAPPA,
                                    lower_bound=DELTA,
                                    boundary_mode="face",
                                    decrease_rate=args.decrease_rate,
                                    mu=eps_M,
                                    eps_M=eps_M,
                                    solver=args.solver,
                                    timelimit=args.timelimit,
                                    solve_options=solve_options,
                                    fallback_solver=None,
                                    do_exact_verify=True,
                                    max_den=max_den,
                                    q_max_den=q_max_den,
                                    screen_tol=args.screen_tol,
                                    psd_fallback=args.psd_fallback,
                                    verify_progress=args.verify_progress,
                                    verify_max_basis=args.verify_max_basis,
                                    verbose=args.verbose,
                                )
                            except Exception as exc:
                                res = error_result(
                                    s, l, n, m, d_L, d_dec_bump, d_bnd_bump,
                                    exc, time.perf_counter() - t0, eps_M,
                                    args.solver)
                                L_exact = None
                                report = None
                                print(f"  CRASHED: {res.solver_status}",
                                      flush=True)
                                stop = True

                            prev_best = best
                            best = better_result(best, res)
                            if best is res and best is not prev_best:
                                best_report = report
                                best_max_den = max_den
                                best_q_max_den = q_max_den
                            if L_exact is not None and res.status == "certified":
                                best = res
                                best_L = L_exact
                                best_report = report
                                best_max_den = max_den
                                best_q_max_den = q_max_den
                                stop = True

                            next_step = escalation_note(
                                d_L_i, bnd_i, dec_i, d_L_values, d_bnd_bumps,
                                d_dec_bumps, max_den_i, max_den_values,
                                best_L is not None)
                            print(attempt_message(
                                res, L_exact, d_dec_bump, d_bnd_bump,
                                next_step), flush=True)
                            if stop:
                                break
                        if stop:
                            break
                    if stop:
                        break
                if stop:
                    break

            exhausted_to_max_d_L = (
                best_L is None
                and not stop
                and max_d_L_tested == max(d_L_values)
            )
            min_lam, max_basis, rough_den = report_denominator_diagnostics(
                best_report, args.screen_tol)
            writer.writerow(result_row(
                best, best_L, args.decrease_rate, best_max_den, best_q_max_den,
                args.mosek_tol, attempts, max_d_L_tested,
                exhausted_to_max_d_L, min_lam, max_basis, rough_den))
            fh.flush()

            if best_L is not None:
                print(f"  -> instance verified; moving on", flush=True)
                done.add(key)
            else:
                print(f"  -> INSTANCE FAILED through d_L={max(d_L_values)} "
                      f"and max_den={max(max_den_values)}", flush=True)

    print(f"\nCSV: {args.csv}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--max-den", type=parse_int_list,
                        default=DEFAULT_MAX_DEN_VALUES,
                        help="Comma-separated rationalization denominators, "
                             "tried in order. mu=eps_M=100/max_den.")
    parser.add_argument("--q-max-den", type=int, default=None,
                        help="Gram rationalization denominator for exact checks. "
                             "Default: max_den*1e4 for each max_den.")
    parser.add_argument("--mosek-tol", type=float, default=1e-10,
                        help="PICOS/MOSEK feasibility and IPM tolerance; <=0 disables.")
    parser.add_argument("--timelimit", type=float, default=900)
    parser.add_argument("--solver", default="mosek")
    parser.add_argument("--decrease-rate", choices=RATE_FORMS, default="L",
                        help="Rate form for the decrease constraint. For q=2, "
                             "'L' avoids the extra p(x) weight.")
    parser.add_argument("--d-L-list", dest="d_L_list", type=parse_int_list,
                        default=DEFAULT_D_L_LIST,
                        help="Comma-separated Lyapunov degrees, tried outermost.")
    parser.add_argument("--d-dec-bumps", type=parse_int_list,
                        default=DEFAULT_D_DEC_BUMPS,
                        help="Comma-separated bumps added to the natural "
                             "decrease multiplier degree floor.")
    parser.add_argument("--d-bnd-bumps", type=parse_int_list,
                        default=DEFAULT_D_BND_BUMPS,
                        help="Comma-separated bumps added to each natural "
                             "boundary multiplier degree floor.")
    parser.add_argument("--m-values", type=parse_int_list, default=DEFAULT_M_VALUES,
                        help="Comma-separated m values, tried outermost in order.")
    parser.add_argument("--s-values", type=parse_int_list, default=(0, 1, 2))
    parser.add_argument("--l-values", type=parse_int_list, default=(1, 2, 3))
    parser.add_argument("--n-values", type=parse_int_list, default=(2, 3, 4))
    parser.add_argument("--screen-tol", type=float, default=1e-9)
    parser.add_argument("--psd-fallback", action="store_true",
                        help="Try SymPy's exact semidefinite PSD fallback "
                             "for borderline direct projections.")
    parser.add_argument("--verify-progress", action="store_true",
                        help="Print compact exact Gram-verification progress.")
    parser.add_argument("--verify-max-basis", type=int, default=30,
                        help="0 disables exact-check size skipping.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not skip instances already present in the CSV.")
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(resume=True)
    return parser


if __name__ == "__main__":
    run_sweep(build_parser().parse_args())

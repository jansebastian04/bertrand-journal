"""Replay archived solver configurations into a separate output directory.

Rows retain their original order, including repeated instances. This replays
the final recorded degrees, not the unrecorded search history. Runtime,
solver outcomes, and numerical candidates may differ between environments.
"""
import argparse
import ast
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/general_lyapunov"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["rescaled", "deflated"], default="rescaled")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--limit", type=int, help="Replay only the first N rows (smoke test)")
    parser.add_argument("--timelimit", type=float, default=900)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    import sympy as sp
    from solver_rescaled import solve_sdp_rescaled, CSV_FIELDS
    from solver_deflated import solve_and_extract_report
    from sweep_deflated import SWEEP_FIELDS, mosek_options, result_row

    filename = ("certificates_rescaled.csv" if args.target == "rescaled"
                else "deflated_sos_basis_v2_m2_q2_verified.csv")
    with (ROOT / "results" / filename).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of submitted results.
    with (args.out_dir / filename).open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS if args.target == "rescaled" else SWEEP_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            print(f"Replaying {args.target} row {index}/{len(rows)}", flush=True)
            kwargs = {key: int(row[key]) for key in ("s", "l", "n", "m", "q", "d_L", "d_dec")}
            kwargs.update(d_bnd=ast.literal_eval(row["d_bnd"]), c=float(row["c"]),
                          lower_bound=sp.Rational(row["delta"]),
                          solver=row["solver"], timelimit=args.timelimit)
            if args.target == "rescaled":
                result = solve_sdp_rescaled(**kwargs, metric="q", tau=float(row["tau"]))
                output = result.row()
            else:
                max_den, q_max_den = int(row["max_den"]), int(row["q_max_den"])
                tol = float(row["mosek_tol"]) if row["mosek_tol"] else None
                result, polynomial, report = solve_and_extract_report(
                    **kwargs, decrease_rate=row["decrease_rate"],
                    mu=float(row["mu"]), eps_M=float(row["eps_M"]),
                    max_den=max_den, q_max_den=q_max_den,
                    solve_options=mosek_options(tol) if tol is not None else None,
                    do_exact_verify=True, psd_fallback=True, verify_max_basis=0,
                )
                output = result_row(result, polynomial, row["decrease_rate"],
                                    max_den, q_max_den, tol, 1, result.d_L, False)
            writer.writerow(output)
            fh.flush()
            print(f"Archived status: {row['status']}; replay status: {result.status}", flush=True)


if __name__ == "__main__":
    main()

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

ARCHIVE = "certificates_rescaled.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--limit", type=int, help="Replay only the first N rows (smoke test)")
    parser.add_argument("--timelimit", type=float, default=900)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    import sympy as sp
    from solver_rescaled import solve_sdp_rescaled, CSV_FIELDS

    with (ROOT / "results" / ARCHIVE).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of submitted results.
    with (args.out_dir / ARCHIVE).open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            print(f"Replaying row {index}/{len(rows)}", flush=True)
            kwargs = {key: int(row[key]) for key in ("s", "l", "n", "m", "q", "d_L", "d_dec")}
            kwargs.update(d_bnd=ast.literal_eval(row["d_bnd"]), c=float(row["c"]),
                          lower_bound=sp.Rational(row["delta"]),
                          solver=row["solver"], timelimit=args.timelimit)
            result = solve_sdp_rescaled(**kwargs, metric="q", tau=float(row["tau"]))
            writer.writerow(result.row())
            fh.flush()
            print(f"Archived status: {row['status']}; replay status: {result.status}", flush=True)


if __name__ == "__main__":
    main()

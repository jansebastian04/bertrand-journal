'Build the rescaled frontier CSV and certificate landscape PDF.\n\nInputs are certificates_rescaled.csv and mstar_delta_grid.csv. No spectral\nwindow screen applies to the rescaled formulation. The retained table uses\nm=2..4 and panels q=0,1,2. The legacy *_m_max_window_open columns report\nthe largest displayed m, not a separate spectral test. --out-dir keeps\nregenerated artifacts separate from the submitted results.\n'
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

RESULTS = Path(__file__).parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--t8", nargs="+", default=["certificates_rescaled.csv"], help="Input certificate CSVs, relative to results")
parser.add_argument("--variants", nargs="+", choices=["q0", "q1", "q2"], default=["q0", "q1", "q2"])
parser.add_argument("--m-max", type=int, default=4)
parser.add_argument("--suffix", default="_rescaled")
parser.add_argument("--out-dir", type=Path, default=RESULTS)
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
VARIANTS = args.variants


def read_csv(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


# ---------- load raw sources ----------
S_KEEP = {"0", "1", "2"}  # Demand exponents covered by the submitted SDP grid.
M_RANGE = list(range(2, args.m_max + 1))

mstar_rows = read_csv("mstar_delta_grid.csv")
minty_mstar = {(r["s"], r["l"], r["n"]): int(float(r["mstar"])) for r in mstar_rows if r["s"] in S_KEEP}
instances = sorted(minty_mstar.keys(), key=lambda k: (int(k[1]), int(k[2]), int(k[0])))  # sort by l, n, s

t8 = [row for name in args.t8 for row in read_csv(name)]
DELTA_LABEL = t8[0]["delta"] if t8 else ""


def variant_of(row):
    return f"q{row['q']}"


sdp = {}
for row in t8:
    key = (row["s"], row["l"], row["n"], row["m"], variant_of(row))
    # Prefer a certified row over any later re-attempt at the same key (e.g. an
    # escalation pass retrying an already-certified instance at a different
    # d_L and landing on "within_margin" instead): a single valid certificate
    # is definitive, so a later non-certified attempt must never hide it.
    # Otherwise last occurrence wins, as before.
    if key in sdp and sdp[key]["status"] == "certified" and row["status"] != "certified":
        continue
    sdp[key] = row


def cell_state(s, l, n, m, variant):
    if s not in S_KEEP or m not in M_RANGE:
        return "no_data"
    key = (s, l, n, str(m), variant)
    status = sdp.get(key, {}).get("status")
    if status == "certified":
        return "certified"
    if status == "error_MemoryError":
        return "crashed"
    if status in ("infeasible", "no_solution", "within_margin"):
        # An attempted search without a certificate is not proof of nonexistence.
        return "no_certificate"
    return "not_attempted"


def cell_d_L(s, l, n, m, variant):
    """Degree d_L of the certified Lyapunov polynomial, or None if this
    cell isn't a certified one (shown as a label inside certified cells)."""
    key = (s, l, n, str(m), variant)
    row = sdp.get(key)
    if row is not None and row["status"] == "certified":
        return int(row["d_L"])
    return None


# ============================================================
# 1. Frontier table: one row per (s,l,n)
# ============================================================
frontier_cols = ["s", "l", "n", "minty_mstar"]
for v in VARIANTS:
    frontier_cols += [f"{v}_m_max_certified", f"{v}_m_max_window_open", f"{v}_gap_reason"]

frontier_rows = []
for (s, l, n) in instances:
    row = {"s": s, "l": l, "n": n, "minty_mstar": minty_mstar[(s, l, n)]}
    for v in VARIANTS:
        states_by_m = {m: cell_state(s, l, n, m, v) for m in M_RANGE}
        open_ms = [m for m, st in states_by_m.items() if st != "no_data" and st != "closed"]
        certified_ms = [m for m, st in states_by_m.items() if st == "certified"]
        m_max_open = max(open_ms) if open_ms else ""
        m_max_cert = max(certified_ms) if certified_ms else ""
        gap_reason = ""
        if m_max_open != "" and (m_max_cert == "" or m_max_cert < m_max_open):
            # first non-certified m above the certified frontier
            floor = m_max_cert if m_max_cert != "" else min(open_ms) - 1
            for m in sorted(open_ms):
                if m > floor:
                    gap_reason = states_by_m[m]
                    break
        row[f"{v}_m_max_certified"] = m_max_cert
        row[f"{v}_m_max_window_open"] = m_max_open
        row[f"{v}_gap_reason"] = gap_reason
    frontier_rows.append(row)

frontier_name = f"frontier_table{args.suffix}.csv"
with open(args.out_dir / frontier_name, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=frontier_cols)
    w.writeheader()
    for row in frontier_rows:
        w.writerow(row)

print(f"wrote {len(frontier_rows)} rows to {frontier_name}")

# ============================================================
# 2. Heatmap grid -- built once, plotted immediately below (no intermediate
# file: see module docstring)
# ============================================================
instance_dicts = [{"s": s, "l": l, "n": n, "mstar": minty_mstar[(s, l, n)]} for s, l, n in instances]
cells = {v: [[cell_state(s, l, n, m, v) for m in M_RANGE] for s, l, n in instances] for v in VARIANTS}
d_L_grid = {v: [[cell_d_L(s, l, n, m, v) for m in M_RANGE] for s, l, n in instances] for v in VARIANTS}

# ============================================================
# 3. Render certificate_landscape{suffix}.pdf. One panel per variant; x =
# instance (s,l,n) grouped first into blocks by l (separated by a blank gap,
# not just a line) and within each block by n then s; y = m. A black
# step-line marks the Minty-failure frontier (m*) per instance, so certified
# (green) cells sitting at/above the line are the headline: Lyapunov
# certificates surviving past where the Minty condition already fails.
# ============================================================
n_inst = len(instance_dicts)
n_m = len(M_RANGE)

# Status palette; missing cells receive a neutral background.
COLOR = {
    "no_data": "#fcfcfb",         # never computed -> recedes into the surface
    "not_attempted": "#fab219",   # open, SDP not (yet) run
    "certified": "#0ca30c",       # SDP found a Lyapunov certificate
    "no_certificate": "#ec835a",  # SDP attempted (schedule exhausted or
                                   # manually stopped), no certificate found
    "crashed": "#d03b3b",         # SDP attempted, solver failure
}
LABEL = {
    "no_data": "not computed",
    "not_attempted": "open, not attempted",
    "certified": "certified (label = degree d_L of the certificate)",
    "no_certificate": "attempted, no certificate found (infeasible / no_solution / within_margin)",
    "crashed": "crashed (solver)",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID_LINE = "#e1e0d9"
GROUP_LINE = "#898781"

VARIANT_TITLE = {
    "q0": "q = 0", "q1": "q = 1", "q2": "q = 2",
}

# ---- x-layout: group instances into 3 blocks by l, with a blank gap between
# blocks (not just a separator line) -- one row = 3 visually distinct blocks ----
BLOCK_GAP = 0.6
l_order = []
for inst in instance_dicts:
    if inst["l"] not in l_order:
        l_order.append(inst["l"])
l_block = {l: idx for idx, l in enumerate(l_order)}


def xpos(i):
    return i + BLOCK_GAP * l_block[instance_dicts[i]["l"]]


right_edge = xpos(n_inst - 1) + 1

# Figure height scales with n_m (rows per panel) so the "n=" / "l=" block
# labels above each panel keep the same relative spacing regardless of how
# many m rows are shown -- fixed at 7.2in per panel for the reference case
# (n_m=4, 3 panels) that the label offsets/fontsizes below were tuned
# against, then scaled by the actual panel count (len(VARIANTS) need not
# be 3 -- one row per requested variant, however many that is).
REF_N_M = 4
fig_height = 7.2 * (n_m + 1.25) / (REF_N_M + 1.25) * len(VARIANTS) / 3

fig, axes = plt.subplots(
    len(VARIANTS), 1, figsize=(13, fig_height), sharex=True,
    gridspec_kw={"hspace": 0.45}, constrained_layout=False, squeeze=False,
)
axes = axes[:, 0]
fig.patch.set_facecolor("#fcfcfb")
fig.subplots_adjust(left=0.035, right=0.995)

for ax, variant in zip(axes, VARIANTS):
    ax.set_facecolor("#fcfcfb")
    grid_cells = cells[variant]  # [instance][m_index] -> state

    for i, inst in enumerate(instance_dicts):
        x = xpos(i)
        for j, m in enumerate(M_RANGE):
            state = grid_cells[i][j]
            ax.add_patch(
                Rectangle(
                    (x, j), 1, 1,
                    facecolor=COLOR[state],
                    edgecolor=GRID_LINE,
                    linewidth=0.6,
                )
            )
            if state == "certified":
                d_L = d_L_grid[variant][i][j]
                if d_L is not None:
                    ax.text(
                        x + 0.5, j + 0.5, str(d_L), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold", zorder=3,
                    )

    def frontier_xy(threshold_per_instance):
        """Step-line coords at the bottom edge of the row = threshold, per
        instance column. A threshold of m_range[-1]+1 still fits: it draws
        flush with the top edge of the grid (holds through every shown m,
        fails just past it). Only a threshold strictly beyond that top edge
        is omitted outright (a gap in the step line), since the true
        frontier there is unknown/outside what's plotted. The line also
        breaks (NaN) at each l-block boundary so it doesn't cut across the
        blank gap between blocks."""
        max_drawable = M_RANGE[-1] + 1
        xs, ys = [], []
        for i, thr in enumerate(threshold_per_instance):
            if i > 0 and instance_dicts[i]["l"] != instance_dicts[i - 1]["l"]:
                xs.append(float("nan"))
                ys.append(float("nan"))
            if thr > max_drawable:
                xs.append(xpos(i) + 0.5)
                ys.append(float("nan"))
                continue
            y = max(0, thr - M_RANGE[0])
            xs += [xpos(i), xpos(i) + 1]
            ys += [y, y]
        return xs, ys

    # Minty-failure frontier: step line at the bottom edge of the m* row
    mstar_vals = [inst["mstar"] for inst in instance_dicts]
    xs, ys = frontier_xy(mstar_vals)
    ax.plot(xs, ys, color=INK, linewidth=1.6, solid_capstyle="butt", zorder=5)

    # (l, n) group separators + labels
    group_bounds = []
    start = 0
    cur = (instance_dicts[0]["l"], instance_dicts[0]["n"])
    for i, inst in enumerate(instance_dicts):
        key = (inst["l"], inst["n"])
        if key != cur:
            group_bounds.append((start, i, cur))
            start = i
            cur = key
    group_bounds.append((start, n_inst, cur))

    for (start, end, (l, n)) in group_bounds:
        ax.axvline(xpos(start), color=GROUP_LINE, linewidth=0.9, zorder=4)
        ax.text(
            (xpos(start) + xpos(end - 1) + 1) / 2, n_m + 0.35, f"n={n}",
            ha="center", va="bottom", fontsize=8.5, color=INK,
        )
    ax.axvline(right_edge, color=GROUP_LINE, linewidth=0.9, zorder=4)

    # bold l-block labels above the n sub-labels
    block_start = 0
    cur_l = instance_dicts[0]["l"]
    for i in list(range(1, n_inst)) + [n_inst]:
        if i == n_inst or instance_dicts[i]["l"] != cur_l:
            ax.text(
                (xpos(block_start) + xpos(i - 1) + 1) / 2, n_m + 0.75, f"l = {cur_l}",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold",
            )
            if i < n_inst:
                block_start = i
                cur_l = instance_dicts[i]["l"]

    ax.set_xlim(0, right_edge)
    ax.set_ylim(0, n_m + 1.25)
    ax.set_xticks([xpos(i) + 0.5 for i in range(n_inst)])
    ax.set_xticklabels([f"s={inst['s']}" for inst in instance_dicts], fontsize=6.5, color=MUTED, rotation=90)
    ax.set_yticks([j + 0.5 for j in range(n_m)])
    ax.set_yticklabels([str(m) for m in M_RANGE], fontsize=8.5, color=INK)
    ax.set_ylabel("m", fontsize=9.5, color=INK, rotation=0, labelpad=12, va="center")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(VARIANT_TITLE[variant], fontsize=11, color=INK, loc="left", fontweight="bold", pad=18)

axes[-1].set_xlabel("instance, grouped by l (blocks), then n, then s", fontsize=9.5, color=INK, labelpad=8)

present_states = {s for v in VARIANTS for row in cells[v] for s in row}

legend_handles, legend_labels = [], []
for k in ["certified", "not_attempted", "no_certificate", "crashed", "no_data"]:
    if k in present_states:
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=COLOR[k], edgecolor=GRID_LINE))
        legend_labels.append(LABEL[k])
legend_handles.append(Line2D([0], [0], color=INK, linewidth=1.6))
legend_labels.append("Minty-failure frontier m* (above: Minty fails)")

fig.legend(
    legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False,
    bbox_to_anchor=(0.5, -0.07), fontsize=9, labelcolor=INK,
)

title = "SDP certificate landscape vs. Minty-failure frontier, by regularizer"
subtitle_bits = []
if DELTA_LABEL:
    subtitle_bits.append(rf"$\delta={DELTA_LABEL}$")
title += "  (" + ", ".join(subtitle_bits) + ")"
fig.suptitle(title, fontsize=13, color=INK, y=1.03, fontweight="bold")

pdf_name = f"certificate_landscape{args.suffix}.pdf"
fig.savefig(args.out_dir / pdf_name, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"wrote {pdf_name}")

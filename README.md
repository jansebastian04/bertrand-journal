# Bayesian Bertrand games: numerical supplement

Code and archived results for SOS Lyapunov certificates and Minty-condition
counterexamples in Bayesian Bertrand competition. Run commands from the repository
root with Python 3.13.

The repository implements one SDP, `solver_rescaled.py`, which computes the
frontier results by searching for floating-point Lyapunov functions. Feasibility
is reported from the solver's own termination status, without exact rational
verification.

## Contents

| Location | Purpose |
| --- | --- |
| `src/bertrand.py` | Game, equilibrium, and symbolic/numerical gradients |
| `src/sos_common.py` | Shared model construction, q-metrics, candidates, SOS constraints |
| `scripts/general_lyapunov/solver_rescaled.py` | Floating-point SDP for the frontier results |
| `scripts/general_lyapunov/plot_lyapunov.py` | Phase portraits and Lyapunov level sets |
| `scripts/geometry/` | Minty threshold grid and numerical tightness search |
| `scripts/reproduce.py` | Replay the solver configurations recorded in the archive |
| `results/build_presentation.py` | Build the frontier table and landscape |
| `results/` | Submitted CSVs and PDFs, preserved unchanged during cleanup |
| `tests/` | Model, degree-selection, and SOS construction regression checks |

Only q=0,1,2 regularizers are retained. Empty `c_floor` and `cond_P` columns
remain in the rescaled CSV schema for compatibility with the archive.

## Environment

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On Linux/macOS, activate with `source .venv/bin/activate`. For runtime dependencies
alone, install `requirements.txt`. The pinned versions record the environment
used to validate this cleanup, not a recovered environment for all historical
runs. The archived solver outputs identify MOSEK; a valid MOSEK license is
required to replay those runs. CVXOPT is retained for the rescaled solver's
existing fallback. A fallback is identified in the output and is not equivalent
to a MOSEK replay.

## Reproduction

Write regenerated artifacts to `reproduced/` to preserve the submitted data for
comparison. The replay driver refuses to overwrite an existing output CSV; use
a new output directory for a fresh replay.

| Archived artifact | Generator |
| --- | --- |
| `mstar_delta_grid.csv` (36 rows) | `scripts/geometry/mstar_delta_grid.py` |
| `mstar_tightness_check.csv` (74 rows) | `scripts/geometry/mstar_tightness_check.py` |
| `certificates_rescaled.csv` (301 rows) | `scripts/reproduce.py` |
| `frontier_table_rescaled.csv`, `certificate_landscape_rescaled.pdf` | `results/build_presentation.py` |
| Nine `results/plots/lyapunov_rescaled_*.pdf` files | `scripts/general_lyapunov/plot_lyapunov.py --from-csv` |

```powershell
python scripts/geometry/mstar_delta_grid.py --out reproduced/mstar_delta_grid.csv
python scripts/geometry/mstar_tightness_check.py --csv reproduced/mstar_delta_grid.csv --out reproduced/mstar_tightness_check.csv
python scripts/reproduce.py
python results/build_presentation.py --out-dir reproduced
python scripts/general_lyapunov/plot_lyapunov.py --from-csv --out-dir reproduced/plots --no-show
```

The last two commands use the **archived** inputs to reconstruct the submitted
presentation. The landscape uses m=2..4 and panels q0, q1, q2. Its legacy
`*_m_max_window_open` columns report the displayed upper m; no spectral-window
test applies to this formulation. Failed finite-degree searches do not prove
that no Lyapunov function exists.

For a presentation using regenerated solver data, pass
`--t8 ../reproduced/certificates_rescaled.csv` to `build_presentation.py`, or
`--from-csv reproduced/certificates_rescaled.csv` to `plot_lyapunov.py`.
The presentation script continues to use the archived Minty threshold grid.

A small solver replay:

```powershell
python scripts/reproduce.py --limit 1 --out-dir reproduced/smoke
```

The replay driver reads the degrees, delta, c, tau, and solver from each
archived row, retaining the current face-boundary formulation. Search history is
not recorded in the archive, so a replay reproduces the accepted configuration
rather than the search that found it.

## Interpretation and limits

The rescaled solver reports numerical SDP feasibility: a run is feasible when
the modelling layer reports primal status `optimal` or `feasible`. An
infeasibility message concerns the fixed template (d_L, d_dec, d_bnd, tau, c)
rather than the dynamics, and any other termination, including a time-limit
cutoff, is inconclusive. The decrease constraint imposes
dL/dt <= -c*r*prod(x)*L, with r the absolute spectral abscissa of
Hinv(x*) Dv(x*) rationalized to denominator at most 10^6; an instance whose
abscissa falls below that resolution receives r=0, and its constraint
degenerates to non-increase of L.

The tightness search combines sampling, differential evolution, SLSQP, and a
grid. Its archived `satisfies` label means **no violation was found**, not a proof
of the universal Minty inequality. Seed 0 is the default. Resuming skips completed
instances and changes subsequent random-number consumption compared with a fresh
run; use a fresh output for a controlled run.

The archive records final solver rows rather than every attempt. The rescaled
CSV does not store its Lyapunov polynomials, so plotting re-solves the SDP.
Runtime, coefficients, feasibility near tolerance boundaries, and PDF bytes can
vary. Full sweeps are expensive; solver time limits do not bound symbolic
construction time.

Cleanup removed the presentation script's unconditional read of a missing
spectral-screen CSV. See `VALIDATION.md` for checks actually completed.

Local environments, caches, editor/agent settings, and `reproduced/` are excluded
by `.gitignore` and are not part of the numerical supplement.

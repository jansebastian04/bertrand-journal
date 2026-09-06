# Bayesian Bertrand games: numerical supplement

Code and archived results for SOS Lyapunov certificates and Minty-condition
counterexamples in Bayesian Bertrand competition. Run commands from the repository
root with Python 3.13.

The repository implements two SDPs:

- **`solver_rescaled.py`** computes the frontier results by searching for
  floating-point Lyapunov functions, without exact verification.
- **`solver_deflated.py`** modifies the rescaled SDP to make exact rational
  verification possible. Currently, only a small subset of instances can be
  verified; these results do not establish verification of the full frontier.

## Contents

| Location | Purpose |
| --- | --- |
| `src/bertrand.py` | Game, equilibrium, and symbolic/numerical gradients |
| `src/sos_common.py` | Shared model construction, q-metrics, candidates, SOS constraints |
| `src/sos_deflation.py` | Centered bases, degree reduction, SDP margin helpers |
| `scripts/general_lyapunov/solver_rescaled.py` | Floating-point SDP for the frontier results |
| `scripts/general_lyapunov/solver_deflated.py` | Modified SDP enabling exact verification |
| `scripts/general_lyapunov/` | Supporting deflated sweep and Lyapunov plots |
| `scripts/geometry/` | Minty threshold grid and numerical tightness search |
| `scripts/reproduce.py` | Replay the solver configurations recorded in the archive |
| `results/build_presentation.py` | Build the frontier table and landscape |
| `results/` | Submitted CSVs and PDFs, preserved unchanged during cleanup |
| `tests/` | Model, exact-arithmetic, and SOS construction regression checks |

The structured solver and older experiments are not required. Exact verification
helpers are defined directly in `solver_deflated.py`. Only q=0,1,2 regularizers
are retained. Empty `c_floor` and `cond_P` columns remain in the rescaled CSV
schema for compatibility with the archive.

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
| `certificates_rescaled.csv` (301 rows) | `scripts/reproduce.py --target rescaled` |
| `deflated_sos_basis_v2_m2_q2_verified.csv` (29 rows) | `scripts/reproduce.py --target deflated` |
| `frontier_table_rescaled.csv`, `certificate_landscape_rescaled.pdf` | `results/build_presentation.py` |
| Nine `results/plots/lyapunov_rescaled_*.pdf` files | `scripts/general_lyapunov/plot_lyapunov.py --from-csv` |

```powershell
python scripts/geometry/mstar_delta_grid.py --out reproduced/mstar_delta_grid.csv
python scripts/geometry/mstar_tightness_check.py --csv reproduced/mstar_delta_grid.csv --out reproduced/mstar_tightness_check.csv
python scripts/reproduce.py --target rescaled
python scripts/reproduce.py --target deflated
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
python scripts/reproduce.py --target rescaled --limit 1 --out-dir reproduced/smoke
```

Fresh deflated searches:

```powershell
python scripts/general_lyapunov/sweep_deflated.py --csv reproduced/sweep_L.csv --decrease-rate L
python scripts/general_lyapunov/sweep_deflated.py --csv reproduced/sweep_pL.csv --decrease-rate pL
```

The default sweep covers m=2, q=2, s=0,1,2, l=1,2,3, n=2,3,4. The archived
deflated CSV contains both `L` and `pL` runs. The replay driver reads the rate,
degrees, denominator bounds, mu, eps_M, delta, c, and MOSEK tolerance from each
row, retaining the current full, relatively scaled basis and face-boundary
formulation. It requests all exact checks without a Gram-basis size cutoff.
Basis choices and some verification options are not recorded in the historical
CSV, so its complete historical configuration cannot be recovered from that
file alone.

## Interpretation and limits

The rescaled solver reports numerical SDP feasibility. The deflated solver also
reconstructs rational Gram matrices, checks coefficient identities, and proves
positive semidefiniteness; only a successful exact pass writes a verified
polynomial. `L` imposes dL/dt <= -c*r*L; `pL` imposes
dL/dt <= -c*r*prod(x)*L. The deflated SDP fixes mu and eps_M and maximizes
the absolute residual Gram floor eps_R.

The tightness search combines sampling, differential evolution, SLSQP, and a
grid. Its archived `satisfies` label means **no violation was found**, not a proof
of the universal Minty inequality. Seed 0 is the default. Resuming skips completed
instances and changes subsequent random-number consumption compared with a fresh
run; use a fresh output for a controlled run.

The archive records final solver rows rather than every attempt. The rescaled
CSV does not store its Lyapunov polynomials, so plotting re-solves the SDP.
Runtime, coefficients, feasibility near tolerance boundaries, and PDF bytes can
vary. Full sweeps are expensive; solver time limits do not bound symbolic
construction or exact verification time.

Cleanup restored the deflated sweep's missing rate-choice interface and removed
the presentation script's unconditional read of a missing spectral-screen CSV.
See `VALIDATION.md` for checks actually completed.

Local environments, caches, editor/agent settings, and `reproduced/` are excluded
by `.gitignore` and are not part of the numerical supplement.

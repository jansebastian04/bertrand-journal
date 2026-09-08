# Cleanup validation

Validated on Windows with Python 3.13.8 and the versions in `requirements.txt`.

- SHA-256 checks confirmed that all 14 archived CSV/PDF artifacts are unchanged.
- The regression suite covers equilibrium identities, candidate vanishing,
  multiplier-degree selection, the rationalized spectral rate, construction of
  the rescaled SDP at its archived variable count, constant and metric
  validation, and all 36 archived analytical Minty thresholds. All 16 checks
  pass.
- Regenerated `frontier_table_rescaled.csv`: all 27 rows match the archive.
- Regenerated `certificate_landscape_rescaled.pdf` successfully using the
  archived CSVs without the missing spectral-screen input.
- Regenerated the full first `mstar_delta_grid.csv` row, including `delta_max`:
  exact CSV field match. The full 36-row delta search was stopped during
  expensive symbolic model construction; it was not completed.
- Replayed the first archived rescaled configuration, s=1, l=1, n=2, m=4,
  q=2, d_L=2, d_dec=6, d_bnd=0: MOSEK returned `optimal`, status `certified`,
  with the archived 6,815 scalar SDP variables.
- Replayed the smallest archived rescaled configuration, s=2, l=1, n=2, m=2,
  q=2, d_L=2, d_dec=4, d_bnd=0: MOSEK returned `optimal`, status `certified`,
  with the archived 199 scalar SDP variables.
- Regenerated the s=2, l=1, n=2, m=2, q=2 Lyapunov PDF from its archived
  rescaled row, using a reduced plotting grid of 40 for the smoke test.

Full solver sweeps, the 74-row stochastic tightness search, and all nine
full-resolution Lyapunov plots were not rerun. Numerical and PDF byte identity
is not claimed. Regenerated check outputs are under ignored `reproduced/`.

Cleanup removed the presentation script's unconditional read of a missing
spectral-screen CSV.

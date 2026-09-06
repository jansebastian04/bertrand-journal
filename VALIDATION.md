# Cleanup validation

Validated on Windows with Python 3.13.8 and the versions in `requirements.txt`.

- SHA-256 checks confirmed that all 15 archived CSV/PDF artifacts are unchanged.
- The regression suite covers equilibrium identities, candidate vanishing,
  definite/singular/indefinite PSD cases, exact affine projection, distinct `L`
  and `pL` rate polynomials, SOS construction, metric validation, and all 36
  archived analytical Minty thresholds.
- Regenerated `frontier_table_rescaled.csv`: all 27 rows match the archive.
- Regenerated `certificate_landscape_rescaled.pdf` successfully using the
  archived CSVs without the missing spectral-screen input.
- Regenerated the full first `mstar_delta_grid.csv` row, including `delta_max`:
  exact CSV field match. The full 36-row delta search was stopped during
  expensive symbolic model construction; it was not completed.
- Replayed the first archived rescaled configuration, s=1, l=1, n=2, m=4,
  q=2, d_L=2, d_dec=6, d_bnd=0: MOSEK returned `optimal`, status `certified`,
  with the archived 6,815 scalar SDP variables.
- Replayed the first archived deflated configuration, s=0, l=1, n=2, m=2,
  q=2, d_L=4, d_dec=10, d_bnd=4, rate `pL`: MOSEK returned `optimal`,
  status `certified`, with 1,114 scalar SDP variables. Exact verification
  passed the candidate lower bound, all nine multiplier Gram matrices,
  and all four residual Gram matrices.
- Regenerated the s=2, l=1, n=2, m=2, q=2 Lyapunov PDF from its archived
  rescaled row, using a reduced plotting grid of 40 for the smoke test.

Full solver sweeps, the 74-row stochastic tightness search, and all nine
full-resolution Lyapunov plots were not rerun. Numerical and PDF byte identity
is not claimed. Regenerated check outputs are under ignored `reproduced/`.

The retained code originally had a broken deflated sweep interface: the sweep
imported an undefined `RATE_FORMS` and supplied `decrease_rate` to a solver
that did not accept it. Both archived choices (`L`, `pL`) now propagate through
degree selection and polynomial construction. The first archived `pL`
configuration passed the replay and exact verification described above.

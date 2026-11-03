# Run-time estimate (written by vof/make_cases.py)

These are estimates. They are meant for planning; the test job (`vof/hpc/test_vof.sbatch`)
measures the real cost per step on Euler, and `vof/analyse_run.py --timing <run>` rescales them.

- **Cells:** octree estimate of the snappyHexMesh refinement (same levels, 2 buffer cells
  between levels). Expect ±30 %; `log.checkMesh` gives the real count.
- **Physical time:** the model's quasi-static drainage curve Q_L(h) through the CFD pool,
  t = ∫ A_fs(h) / Q_L(h) dh, from the start level until Q_L < 10 mL/min, plus the
  0.3 s stop window (and 0.3 s of start-up for the short start).
- **Time step:** maxCo = maxAlphaCo = 1. Once air reaches the bore, it limits the step:
  the air jets through the bore at sqrt(2 (p_g - p_s) / ρ_air) ≈ 30–40 m/s, on bore cells of
  0.2 mm (coarse) or 0.1 mm (fine). Before that, the capillary limit (maxDeltaT) and the
  liquid velocity set it.
- **Cost per step:** 10 µs per cell per core (interFoam, PIMPLE with 3 pressure correctors,
  GAMG), 48 cores at 80 % parallel efficiency. This is the number to calibrate with the test job.
- **Air ×10:** air density 12.04 kg/m³ instead of 1.204. Both resistances in series, the exposed
  holes and the bore, scale with ρ_air, so the vacuum in the drain and the stop level do not change.
  The air velocities drop by √10, and the time step rises by about 3. This is a numerical
  shortcut; use it for the long runs if real air is too expensive.

| Case | Start | Mesh | Air density | Cells | Physical time (s) | Two-phase part (s) | Δt two-phase (µs) | Time steps | s/step (48 cores) | Wall time (days) | Core-hours |
|---|---|---|---|---|---|---|---|---|---|---|---|
| usz_tilt10 | full (33.2 mm) | coarse | ×1 | 1.34 M | 7.4 | 6.6 | 5.2 | 1.30 M | 0.35 | 5.3 | 6,050 |
| usz_tilt10 | full (33.2 mm) | coarse | ×10 | 1.34 M | 7.4 | 6.6 | 16.4 | 0.42 M | 0.35 | 1.7 | 1,959 |
| usz_tilt10 | short (25.0 mm) | coarse | ×1 | 1.34 M | 5.6 | 5.6 | 5.2 | 1.09 M | 0.35 | 4.4 | 5,068 |
| usz_tilt10 | short (25.0 mm) | coarse | ×10 | 1.34 M | 5.6 | 5.6 | 16.4 | 0.34 M | 0.35 | 1.4 | 1,603 |
| usz_tilt10 | full (33.2 mm) | fine | ×1 | 4.43 M | 7.4 | 6.6 | 2.6 | 2.61 M | 1.15 | 34.8 | 40,097 |
| usz_tilt10 | full (33.2 mm) | fine | ×10 | 4.43 M | 7.4 | 6.6 | 8.2 | 0.85 M | 1.15 | 11.4 | 13,105 |
| usz_tilt10 | short (25.0 mm) | fine | ×1 | 4.43 M | 5.6 | 5.6 | 2.6 | 2.18 M | 1.15 | 29.0 | 33,433 |
| usz_tilt10 | short (25.0 mm) | fine | ×10 | 4.43 M | 5.6 | 5.6 | 8.2 | 0.69 M | 1.15 | 9.2 | 10,572 |
| rec_tilt10 | full (33.2 mm) | coarse | ×1 | 0.78 M | 13.4 | 9.7 | 6.8 | 1.48 M | 0.20 | 3.5 | 4,026 |
| rec_tilt10 | full (33.2 mm) | coarse | ×10 | 0.78 M | 13.4 | 9.7 | 21.5 | 0.52 M | 0.20 | 1.2 | 1,409 |
| rec_tilt10 | short (6.4 mm) | coarse | ×1 | 0.78 M | 5.8 | 5.8 | 6.8 | 0.85 M | 0.20 | 2.0 | 2,292 |
| rec_tilt10 | short (6.4 mm) | coarse | ×10 | 0.78 M | 5.8 | 5.8 | 21.5 | 0.27 M | 0.20 | 0.6 | 725 |
| rec_tilt10 | full (33.2 mm) | fine | ×1 | 2.65 M | 13.4 | 9.7 | 3.4 | 3.03 M | 0.69 | 24.2 | 27,925 |
| rec_tilt10 | full (33.2 mm) | fine | ×10 | 2.65 M | 13.4 | 9.7 | 10.7 | 1.10 M | 0.69 | 8.8 | 10,139 |
| rec_tilt10 | short (6.4 mm) | fine | ×1 | 2.65 M | 5.8 | 5.8 | 3.4 | 1.69 M | 0.69 | 13.5 | 15,579 |
| rec_tilt10 | short (6.4 mm) | fine | ×10 | 2.65 M | 5.8 | 5.8 | 10.7 | 0.53 M | 0.69 | 4.3 | 4,927 |

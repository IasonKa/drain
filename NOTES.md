# Project notes

Written 16 November 2026, when the local working copy was removed. Everything needed to pick the
project up again is in this repository.

## Where things are

- Branch `main`: `drainsponge/`, `drainstop/`, `visualization/`.
- Branch `Irrigation-for-Weekend`: all of the above, plus `usz_no_inner_tube/`, `bibliography/`,
  the archive zips and this file.
- `DrAinFebio.zip` is over GitHub's 100 MB limit and is stored in two parts. Rebuild it with
  `cat DrAinFebio.zip.part00 DrAinFebio.zip.part01 > DrAinFebio.zip`.

## Study 1: drainsponge (sponge tissue on one hole, OpenFOAM + FEBio)

- Models a biphasic sponge tissue on one hole of the drain. Question: does tissue get sucked into the
  holes, and how much liquid drains.
- Designs: `current` (dome cap, plain hole); `standoff` (ring r 0.55-0.85 mm, 0.40 mm high, 4 slots of
  0.35 mm, measured from the STL); `standoff_opt` (0.50 mm high, r_out 1.00 mm, slots 0.50 mm).
- Scenarios: film edge open (free liquid) or closed (tissue liquid only).
- OpenFOAM and FEBio were never run. All 3D runs are untested. The figures and videos come from an
  axisymmetric preview model with placeholder inputs (E 10 kPa, 10 kPa suction).
- Euler Slurm scripts are in `hpc/`: stack/2024-06, openfoam/2306, FEBio v4.13 built from source.
  Shell and Slurm scripts must keep LF line endings and the executable bit.

## Study 2: drainstop (two-phase CFD with Kratos, closed 5 November 2026)

- `usz_tilt10_h21_coarse`: valid to 15 ms, diverged at 16.4 ms when a liquid slug reached the outlet
  boundary. The outlet carries mainly air and a trickle of liquid, the near-stop state the model predicts.
- `rec_tilt10_h8`: 50 ms, not settled. The drain started 94 % full and was siphoning out its starting
  liquid. Not comparable with the model's steady value.
- Results: `drainstop/vof/kratos/results/`; paper Section 8. The run folders were never in the repo.

## Study 3: usz_no_inner_tube (irrigation flow, November 2026)

- Design: internal tube removed, V funnel to the suction tube, all side holes kept, irrigation outlet in
  the tip pointing at the suction.
- Conditions: water only (no air, tissue, wall movement or clogging), irrigation 250 mL/h, suction
  -7 mmHg, outlet 1 mm (also 2 mm). All flows laminar.
- Blocked sheath: the jet makes a ring vortex 38 mm long that recirculates 10 mL/min, 2.4 times the
  irrigation. In steady laminar flow the vortex traps its fluid instead of mixing it.
- Open holes (four at a time): every hole draws about 38 mL/min inwards. No irrigant leaves through
  the holes.
- Best outlet position: the tip. Old fluid left after 10 min with a 1 mm outlet: 31 % at the tip,
  38 % at 15 mm, 49 % at 30 mm, 55 % at 45 mm, 64 % at 60 mm, 78 % at 80 mm, 88 % at 100 mm.
- A 2 mm outlet is slightly better: 27 % at the tip, 34 % at 15 mm.
- Vortex length: x_end = Re (0.078 R_eq - d/16), with R_eq = 5.45 mm. Gives 32 mm for d = 1 mm
  (simulation 38 mm) and 13 mm for d = 2 mm.
- Reports: `usz_no_inner_tube/paper/final_report.pdf` and `jet_loop_maths.pdf`.

## Open points

- The page layout of the two PDFs in `usz_no_inner_tube/paper/` was never checked by eye.
- Open-hole runs are short (0.43 s of physical time) and coarse; the inflow balance is off by 22 %.
- The 1 mm sweep runs were stopped while still changing about 0.2 % per step.
- Not simulated: pulsed irrigation, higher irrigation rate, stronger suction, a position sweep with
  holes open.
- Files from an older stub design are still in `usz_no_inner_tube/`: `render.py`, pictures 01-04,
  `usz_sheath_no_inner_tube.stl`, `flow/setup.png`.

## Lost when the local copy was removed (never in this repository)

- STEP CAD files, COMSOL notes and CAD scripts from a local `others` folder.
- Solver input meshes (`flow.mdpa`), Kratos VTK outputs and failed trial runs. These can be regenerated.
- Two source STLs in `drainsponge/data` (`drainage-3D_cap_dome.stl`, `drainage-3D_cap_dome_standoff.stl`).

## How to rerun study 3

Python 3.12, then

    pip install KratosMultiphysics==10.3.1 KratosFluidDynamicsApplication==10.3.1 KratosLinearSolversApplication==10.3.1 gmsh==4.15.2 pyvista==0.49.2 numpy matplotlib imageio

In `usz_no_inner_tube/flow`:

    python make_flow_case.py blocked --s <mm from tip> [--d <outlet mm>] --name <case>
    python run_flow.py runs/<case> --threads 12 --steps 55 --dt0 0.02 --grow 1.3 --dtmax 5 --tol 1e-4
    python flush_score.py runs/<case>
    python plot_sweep.py

One flow run takes about 35 minutes on a 12-thread laptop CPU. The reports are built with Tectonic:
`tectonic final_report.tex`.

## Working preferences

- Reports as a compiled PDF paper with the mathematics, drawings and graphs.
- Each study folder is self-contained: PDF, results, code, and how to rerun.
- Push to GitHub after every update.
- Short, plain explanations.

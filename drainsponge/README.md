# drainsponge: sponge-like (biphasic) tissue on one drain hole, OpenFOAM + FEBio

This package models one hole of the drain with a piece of soft, liquid-filled tissue lying on it, then asks what happens when the suction is switched on:

- Does the tissue get sucked into the hole and block it?
- How much liquid still reaches the hole: free wound liquid, and liquid drawn out of the tissue itself?

Pictures and videos of everything below are in [`../visualization/`](../visualization/). Open `index.html` there.

## Designs compared (`drainsponge/config.py`, `DESIGNS`)

| Name | What it is |
|---|---|
| `current` | Dome-cap drain (`drainage-3D_cap_dome`): plain Ø1.0 mm hole in the 0.6 mm wall. The tissue starts 50 µm above the skin. |
| `standoff` | The new offset design (`new_offset_desing.zip`). A ring around every hole: r 0.55–0.85 mm, 0.40 mm high, cut by 4 parallel slots 0.35 mm wide (all measured from the STL). The tissue starts 50 µm above the ring tops. |
| `standoff_opt` | Suggestion from the ring screening: 0.50 mm high, r 0.55–1.00 mm, 0.50 mm slots. |

## Scenarios

- **Free liquid around (`film_edge: open`)**: the gap under the tissue is open to a pool of wound liquid.
- **No free liquid (`film_edge: closed`)**: all liquid comes out of the tissue. This is the case that decides whether tissue blocks the hole.

## Physics

| Part | What is modelled |
|---|---|
| Liquid (channel, hole, slots, gap) | OpenFOAM, steady laminar Navier–Stokes, inertia included |
| Tissue | FEBio **biphasic**: neo-Hookean solid matrix + liquid in the pores, Darcy flow, optional strain-dependent permeability, large deformation, transient |
| Tissue ↔ liquid | pore pressure = film pressure, film pressure as traction, seepage in/out, squeeze-film motion |
| Tissue ↔ drain | contact with friction on the skin, ring tops and walls, and hole edge; a 3 µm residual film stands for surface roughness |
| Boundaries | tissue top held at body pressure; suction ramped on in 10 ms in the drain channel |

## Package layout

```
drainsponge/   geom.py (structured hex mesh shared by both solvers, incl. ring + slots), foam_mesh.py,
               foam_case.py (OpenFOAM cases), febio_model.py (FEBio model + restart files),
               couple.py (two-way coupling, window by window, resumable), preview.py (fast model),
               post.py (VTU snapshots), cad.py (STEP), config.py (designs, resolutions, inputs)
scripts/       build_cad.py, build_cases.py, run_preview.py, optimize_standoff.py, render_cad.py,
               make_html.py, make_media.py, make_replay.py
hpc/           ETH Euler: setup, FEBio 4.13 build, test job, main coupled job (chains 24 h jobs),
               OpenFOAM-only / FEBio-only / screening jobs, archiving  -> hpc/README_EULER.md
configs/       compare.json (inputs of the comparison shown in the pictures)
tests/         test_coupling.py with mock solvers (restart chaining, resume, fallback)
data/          hole_center.json (+ the drain STLs for rendering, not in git: copy them from the zips)
out/           cad/ (STEP of all designs), optim/ (ring screening results)
```

## How to run

```bash
pip install -r requirements.txt
python scripts/build_cad.py                                          # STEP of the window and the organ, all designs
python scripts/run_preview.py --config configs/compare.json --compare  # fast model, 3 designs x 2 scenarios
python scripts/optimize_standoff.py                                  # ring screening (46 designs)
python scripts/build_cases.py --design standoff --film-edge closed   # OpenFOAM + FEBio inputs
python scripts/render_cad.py && python scripts/make_html.py && python scripts/make_media.py all
python tests/test_coupling.py                                        # coupler test with mock solvers
```

To run the full 3D OpenFOAM ⇄ FEBio coupling on ETH Euler, see **`hpc/README_EULER.md`**: `sbatch hpc/run_coupled.sbatch hpc/production_standoff.json`.

## What the fast model says (E = 10 kPa, suction 10 kPa, 6 h)

| | Current | Standoff (new) | Optimized ring |
|---|---|---|---|
| No free liquid: lowest tissue point vs hole edge | **−283 µm (inside the hole)** | +46 µm | +176 µm |
| Free liquid: lowest tissue point vs hole edge | −350 µm | +425 µm | +537 µm |
| Free liquid: flow into the hole | 0.015 mL/min (sealed) | ~120 mL/min (open) | ~125 mL/min |
| No free liquid: liquid out of the tissue | 3e-5 mL/min | 3e-5 mL/min | 3e-5 mL/min |

In short:

- The plain hole seals within milliseconds, and the tissue bulges into it.
- The ring keeps the tissue out of the hole and the hole open to free liquid.
- How much liquid the tissue itself gives is set by the tissue, not by the drain shape.

## Status and limits

- **Run and checked here:**
  - meshes: closed, positive volumes, volumes match hand calculations
  - CAD
  - the fast model: mass balance to 1e-12, matches 1D consolidation theory
  - the coupler with mock solvers
  - FEBio input/restart syntax, checked against the FEBio 4.13 source
- **Not run yet:** OpenFOAM, FEBio, and the two-way coupling (no solvers on the development PC). These run on Euler.
- **Placeholder inputs:** tissue stiffness, permeability, suction, residual film. Replace them with clinical values.
- **Fast model limits:** it is round, small-deformation and frictionless, and the ring slots are treated as flow ducts. Tissue sagging into the slots is only captured by the 3D model.

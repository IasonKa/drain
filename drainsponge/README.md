# drainsponge: sponge-like (biphasic) tissue on one drain hole, OpenFOAM + FEBio

This package models one hole of the current drain design (dome-cap version of `drainage-3D`,
Ø1.0 mm sharp hole in the 0.6 mm elliptic outer wall, case `hole_center`). A piece of soft,
liquid-filled tissue lies 50 µm above the skin. Suction in the sheath channel then does three things:

1. It draws the **free liquid** in the 50 µm film into the hole. The low film pressure pulls the tissue toward the drain.
2. The tissue **seals onto the hole rim** like a suction cup, and free flow stops. Only a small leak through the roughness film remains.
3. The hole then pulls **liquid out of the tissue itself** (Darcy flow through the pores). Meanwhile the tissue consolidates and bulges into the hole.

The tissue is a FEBio **biphasic** material: a solid matrix plus interstitial liquid. On the OpenFOAM side it is a porous zone (Darcy–Brinkman) or a moving, seeping boundary.

```
drainsponge/            python package
  geom.py               window + structured hex mesh (channel | hole | film | tissue), shared by both solvers
  foam_mesh.py          writes the OpenFOAM polyMesh directly (+ checkMesh-like validation)
  foam_case.py          OpenFOAM cases: "porous" (one-domain Darcy-Brinkman) and "fluid" (for coupling)
  febio_model.py        FEBio 4 biphasic tissue + rigid drain + sliding contact + film-pressure loading
  couple.py             two-way OpenFOAM <-> FEBio coupling driver (Aitken relaxation)
  preview.py            reduced-order axisymmetric model of the same physics (Python, minutes)
  cad.py                exact CAD (STEP) of the window parts and of the sponge organ on the drain
  config.py             all parameters (placeholders, see below)
scripts/
  build_cases.py        -> out/openfoam/*, out/febio/oneway/, out/mesh/*.vtu
  run_preview.py        -> out/preview/preview.json, figures 05-07 (--sweep for 07)
  render_cad.py         -> figures 01-04
  make_html.py          -> out/drainsponge_viewer.html (interactive)
data/                   hole_center.json (frame of the hole in the drain); put drainage-3D_cap_dome.stl
                        (from DrAinFebio.zip, not included - 35 MB) here for render_cad.py
out/                    generated results (CAD, cases, meshes, figures, viewer)
```

## How to run

```bash
pip install -r requirements.txt
python scripts/run_preview.py --sweep        # reduced-order model + figures 05-07 (~10 min with the sweep)
python scripts/build_cases.py                # OpenFOAM cases, FEBio one-way model, meshes
python -c "from drainsponge import cad; from drainsponge.geom import Window; import json; w=Window(); cad.write_window(w,'out/cad'); cad.write_organ(w, json.load(open('data/hole_center.json')), 'out/cad')"
python scripts/render_cad.py                 # figures 01-04 (needs data/drainage-3D_cap_dome.stl)
python scripts/make_html.py                  # out/drainsponge_viewer.html

# on Linux / WSL / cluster with OpenFOAM (ESI) and FEBio 4:
cd out/openfoam/porous_initial && ./Allrun   # rigid sponge, open film
cd out/febio/oneway && febio4 -i tissue.feb  # biphasic tissue, one-way loading
python -m drainsponge.couple --out runs/coupled   # full two-way coupling
```

On Windows, the simplest route to OpenFOAM is the ESI Docker image or WSL (Ubuntu + `openfoam2406`). FEBio 4 has native Windows builds (febio.org), and `couple.py` calls `febio4` from the same shell as OpenFOAM.

## Geometry (local hole frame; `out/cad/`)

| Part | File | What it is |
|---|---|---|
| drain wall | `window_drain.step` | measured elliptic skin 8.925 × 4.835 mm, wall 0.6 mm, Ø1.0 hole (optional fillet) |
| liquid film + hole | `window_film.step` | 50 µm film between skin and tissue + liquid in the hole |
| sheath channel | `window_channel.step` | 1.0 mm deep channel under the wall; suction outlet at y = +1.6 mm |
| tissue | `window_tissue.step` | 3.2 × 3.2 × 2 mm sponge block, its underside follows the skin at +50 µm |
| all of the above | `window_assembly.step` | |
| organ on the drain | `organ_sponge_global.step` | lobulated sponge organ (22 × 30 × 9 mm) **in the drain's global coordinates**: open it together with `drainage-3D_cap_dome.step` |
| window in the drain | `window_film_tissue_global.step` | film + tissue window placed at the hole, global coordinates |

The local frame is the same as drainfeb (`hole_center.json`): origin on the skin at the hole axis, X lateral, Y along the drain, Z along the hole axis. The hole sits at lateral −0.786 mm, so the skin is tilted about 2.7° relative to the hole, as in the real part.

The tissue CAD is a homogeneous solid on purpose. A biphasic material is the homogenised sponge: the pores are in the material model (porosity and permeability), not in the geometry. The pore pattern in the pictures is only a texture.

## The shared mesh

One structured hex mesh covers the window: an O-grid around the hole in plan, then layers channel → wall (fluid only inside the hole) → film → tissue. The default size is 77k cells, with max non-orthogonality 42° and all cells closed and positive.

- OpenFOAM gets the polyMesh written directly, so no gmshToFoam or blockMesh is needed.
- FEBio gets the tissue part as hex8 elements.
- The tissue bottom and the film top are **the same nodes**.
- A deformed tissue goes back into the fluid mesh by moving film and hole nodes vertically only. The topology stays the same, so OpenFOAM can restart from its previous solution. This was checked with the tissue sucked 0.3–0.6 mm into the hole.

## OpenFOAM cases (`out/openfoam/`)

| Case | Content | Run with |
|---|---|---|
| `porous_initial` | tissue = porous cellZone (Darcy d = 1/(k·µ)), 50 µm open film, rigid | `./Allrun` (ESI edition, `simpleFoam` + `fvOptions`) |
| `porous_sealed` | same, tissue shape after 1 h from the preview model (sealed, bulging into the hole) | `./Allrun` |
| `fluid_template` | channel + hole + film only; tissue = moving/seeping wall `tissue_iface` | used by `couple.py`; runs on ESI and Foundation editions |

Every case uses SI units, steady laminar flow, and suction as a fixed (kinematic) pressure at `outlet`. The film edge is open to liquid at p = 0, and the tissue top and sides are at body pressure 0. Flow rates are written to `postProcessing/`:

- `flow_outlet`: total flow
- `flow_film_edge`: free liquid
- `flow_out_of_tissue` (porous cases, via a faceZone): liquid drawn out of the sponge

## FEBio model (`out/febio/oneway/tissue.feb`, `febio_model.py`)

- **Tissue.** `biphasic` material: neo-Hookean matrix plus constant permeability (Holmes–Mow optional). Analysis is TRANSIENT.
- **Drain.** Rigid quad4 shells (skin + hole wall), fixed in place.
- **Contact.** `sliding-elastic` with friction. The contact block is copied from the drainfeb pilot model, which FEBio 4.13 ran.
- **Boundaries.**
  - Top: held in place, p = 0.
  - Sides: symmetry, p = 0.
  - Bottom: the film acts on it as pressure continuity (`prescribed fluid pressure`) plus the matching traction (`pressure` load).
- **Coupling windows.** One FEBio step per coupling window. Nodal maps give the pressure in space; within a window it is interpolated linearly in time (a relative prescribed-pressure map plus two fading pressure-load maps).
- **One-way file.** `oneway/tissue.feb` is loaded with the preview model's film pressures (9 windows, up to 1 h), so it can be run straight away: `febio4 -i tissue.feb`.

## Two-way coupling (`python -m drainsponge.couple --out runs/coupled`)

For each coupling window (log-spaced, 1 ms → 1 h), the driver repeats until converged:

1. FEBio runs 0 → t_n.
2. The tissue bottom positions and seepage flux are read back.
3. The fluid mesh is moved, and the interface velocity (surface motion + seepage) is written into `0/U`.
4. OpenFOAM runs.
5. The interface pressure is read back and relaxed with Aitken.

It needs `febio4` and a sourced OpenFOAM in the same shell (Linux / WSL / cluster). `--dry-run` writes the first inputs only.

## Reduced-order model (`preview.py`) and what it says

This is an axisymmetric version of the same physics, solved monolithically in Python in about 40 s:

- linear Biot poroelastic tissue (Taylor–Hood elements)
- lubrication film (with squeeze and seepage) on the tissue's underside
- hole resistance (Poiseuille + Sampson)
- penalty contact with a 3 µm residual film (roughness)

It was verified against the 1D consolidation (Terzaghi) solution: the steady state matches exactly and the transient is within about 5 %. Mass balance closes to 1e-12.

Default case (E = 20 kPa, ν = 0.2, k = 1e-14 m⁴/(N·s), suction 5 kPa, film 50 µm):

| Quantity | Value |
|---|---|
| peak flow into the hole (free film liquid) | ~2 mL/min, for a few ms |
| film seals on the rim after | ~8 ms |
| flow after sealing (leak under the seal) | ~0.1 → 0.035 mL/min |
| liquid coming out of the sponge itself (1 h) | ~2e-6 mL/min (scales with k) |
| tissue depth into the 1 mm hole | 55 µm at sealing → 67 µm after 1 h |

From `07_parameter_map.png`:

- The depth into the hole scales with suction/E.
- Soft tissue (5–10 kPa) at 10–20 kPa suction is pulled through the whole wall into the channel. That is the occlusion case, and the 3D model should be used there because it is large deformation with contact.
- Stiff tissue at low suction never closes the film.

## Placeholder inputs (to replace with clinical/literature values)

| Input | Default |
|---|---|
| suction | 5 kPa (≈ −38 mmHg) |
| liquid viscosity | 1.5 mPa·s |
| tissue E | 20 kPa |
| tissue ν | 0.2 |
| solid fraction | 0.2 |
| permeability | 1e-14 m⁴/(N·s) |
| friction | 0.2 |
| initial film | 50 µm |
| residual film under the seal | 3 µm |

Change them in a JSON file and pass it with `--config`. For example, `{"tissue": {"E_Pa": 5000}, "suction": {"dp_Pa": 10000}}`.

## Status: what was run and what was not

- **Run and checked here:**
  - CAD: volumes match the analytic values.
  - Meshes: closed, positive volumes, valid face order.
  - The preview model: verified as above.
  - All case files were generated, and the FEBio XML parses.
  - The coupling driver's readers and mappers were checked with synthetic solver output.
- **Not run (no OpenFOAM or FEBio on this machine):**
  - the OpenFOAM cases
  - the FEBio model
  - the two-way loop

  Expect to adjust solver settings on the first real run. Settings worth checking first:
  - the FEBio biphasic keywords (`prescribed fluid pressure` with `type="map"` and `relative`, logfile variables `wx;wy;wz`)
  - the OpenFOAM porous-zone relaxation (`fvSolution`)

## Simplifications to keep in mind

- **Channel.** The window's channel is a simple 1 mm deep, 3.2 mm wide box. In the real drain this hole opens into a narrower sheath channel between ribs (see `04_section_at_hole.png`). This matters only if the tissue gets close to the channel.
- **Neighbouring holes.** These are not in the window, which ends 0.15 mm before the next holes. Their effect is approximated by the open film edge.
- **Preview model.** It is small-strain and frictionless, with a round window and a flat skin. Values above about 200 µm depth are indicative only.

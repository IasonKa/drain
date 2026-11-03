# drainstop: when does the drain stop draining?

This folder studies the two-phase hydraulics of the USZ-scale drain lying in a pool of glycerol-water at a regulated suction of −7 mmHg. It works out the pool level at which the drain stops removing liquid, for a given hole layout, hole size and tilt. It then finds the layout that keeps the drain working longest.

**Read first:** [`paper/drain_stop.pdf`](paper/drain_stop.pdf) (18 pages). It contains the mathematics, drawings of the drain and of every hole layout tested, all graphs, the validation, and the recommended design.

## The question

The drain is a perforated tube lying in a "lake" of liquid with a viscosity between that of water and glycerol:

- the side wall has holes;
- the tip is closed;
- the outlet is a Ø4.24 mm bore centred in the right end face;
- suction is applied at the outlet.

As the lake empties, part of the drain is in air. The drain is not always level.

The questions were:
1. For a given suction of −7 mmHg, at what pool level does the drain stop draining, for a chosen uniform hole layout and hole size (about 0.5 to 4 mm)?
2. Which hole distribution and size keeps it draining longest, so it leaves the least liquid behind?
3. Draw every layout tested, and confirm the answer with simulations.

## Main result

Air drawn in through exposed holes, in series with the outlet bore, leaves a vacuum `|p_s| / (1 + Λ²)` inside the drain. Λ is the exposed open area divided by the bore area, each weighted by its discharge coefficient. Liquid can only leave through the part of the bore below the inner liquid level, so drainage stops when

    h + H_v / (1 + Λ_w(h)²) = z_lip        H_v = |p_s| / (ρ g) = 84.5 mm

or when no hole is left under the surface. Λ_w is Λ refined for air that must enter against liquid standing inside the drain.

The table gives the pool level left when the drain stops:

| Design | Flat | Outlet 10° up | Outlet 20° up |
|---|---|---|---|
| USZ as built (318 × Ø1.0 through both faces, 468 mm²) | 2.2 mm | 20.0 mm | 36.9 mm |
| Recommended: 2 staggered rows at ±30°, 36 × Ø1.0 (28 mm²) | 0.5 mm | 1.4 mm | 2.1 mm |

Why the recommended design works:
- **Open area.** Keep the total open area at 1–2 times the bore area. More area adds almost no flow, because the bore already limits it, but it destroys the vacuum once holes are exposed. The as-built field has about 33 times the bore area.
- **Hole size.** Size matters only through the total open area and how evenly the holes cover the length. 1.0–1.5 mm gives enough positions to reach both ends.
- **Position.** Put the rows on the lower half (about 5 and 7 o'clock), staggered by half a pitch.
- **Tilt.** With the outlet raised, the lip the liquid must reach rises by `L sin α`, so tilt is the critical case. The recommended layout drains down to its lowest hole edge for tilts up to 22.5° with the outlet up.

### Suction source (decision: wall regulator)

The drain is meant for a wall regulator. The regulator must hold −7 mmHg while passing this much air (`results/mincap.json`):

| Tilt (outlet up) | 0° | +5° | +10° | +15° | +20° |
|---|---|---|---|---|---|
| Air flow needed | any | 9 L/min | 12 L/min | 15 L/min | 18 L/min |

Once fully exposed, the drain draws 19–23 L/min of air. A source with less capacity needs fewer holes (Table 5 in the paper):

| Source capacity | Holes at Ø1.0 |
|---|---|
| ~5 L/min | ~12 |
| ~10 L/min | ~24 |

## Pictures of the simulation

The [`pictures/`](pictures) folder shows the drain while the pool empties, with the outlet end 10° up and the as-built USZ field next to the recommended layout:

| File | Moment |
|---|---|
| `drain_emptying_tilt10.gif` | the whole drainage, about 12 s of animation for 165 s of draining |
| `tilt10_1_submerged.png` | start: both drains fully submerged and full of liquid |
| `tilt10_2_air_enters.png` | the first exposed openings of the USZ part start to draw air |
| `tilt10_3_usz_stops.png` | the USZ part stops at 20 mm: about 28 L/min of air through its exposed openings, no liquid |
| `tilt10_4_end.png` | the recommended part has drained to its lowest openings (1.4 mm) |

Each picture shows three things:
- **Side view:** a cut through the drain axis showing the pool, the drain wall, the liquid and air inside the drain and the outlet tube. Each opening on the near side is coloured by what flows through it: liquid in (blue), air in (orange) or nothing (grey).
- **3D view from below:** the wetted skin and every opening, coloured the same way.
- **Level plot:** the pool level against time for both designs.

These are pictures of the network simulation's state: the flow through every opening, the pressure and the liquid level inside the drain. They are not CFD flow fields; for those, see `vof/`. To make them for another tilt, run `python scripts/make_pictures.py 15` (about 4 minutes).

## How it was done, and why

1. **Hole and bore hydraulics.** Each hole obeys `Δp = R_v q + K q|q|`:
   - `R_v` is the viscous resistance (Sampson orifice plus Poiseuille through the 0.6 mm wall).
   - `K` is the inertial loss, using discharge coefficients from Lichtarowicz.
   - The bore uses a sudden-contraction coefficient of 0.85.

   Both terms are needed because, at 0.5–4 mm holes in 6 mPa·s liquid, the flow is neither fully viscous nor fully inertial.
2. **Capillarity check.** At −7 mmHg, only holes below 0.29 mm could hold air out by surface tension. So every hole in the requested range lets air in once exposed, and the problem is a two-phase pressure divider, not a capillary seal.
3. **Closed-form stop condition** (above, derived in Section 3 of the paper). It gives the answer in one line and shows which parameter matters: the exposed open area relative to the bore.
4. **Quasi-static two-phase network simulation** (`drainstop/model.py`):
   - Every hole is split into 61 area elements, so partly submerged holes are handled.
   - Liquid and air flows through all holes and the bore are solved together, with a separated-phase bore (liquid below the inner level, gas above).
   - The pool is lowered in 0.1 mm steps until the liquid flow drops below 0.1 mL/min.

   This checks the closed form without its simplifications.
5. **Time-dependent simulation** (`drainstop/transient.py`). It is an independent model with liquid storage inside the drain and a rigid-volume gas pressure, stepped in time. It confirms that the quasi-static picture holds while the pool is actually emptying.
6. **Studies** (all in `scripts/run_study.py`):
   - `sweep`: 1127 cases (11 layouts × 8 hole sizes × 7 tilts). The closed form and the network simulation agree within 0.03 mm on average.
   - `curves`: liquid and air flow against pool level.
   - `optimize`: hole size × count map for two rows at ±30°.
   - `angles`: which row angle is best.
   - `sensitivity`: one-at-a-time changes to viscosity, suction, discharge coefficients, capillarity and drawdown.
   - `montecarlo`: 200 random operating conditions per design.
   - `transient`: time-dependent validation.
   - `tiltlimit`: tilt from −15° to +40°.
   - `pump`: limited source capacity.
   - `mincap`: minimum regulator flow for the recommended design.
   - `sizes_rec`: every hole size at the recommended layout.
7. **3D CFD check, prepared but not yet run** ([`vof/`](vof/README.md)). There is no OpenFOAM on this PC. Two OpenFOAM `interFoam` volume-of-fluid cases are set up instead for the ETH Euler cluster, both with the outlet 10° up:
   - the as-built USZ field, where the model predicts a stop at 20.0 mm;
   - the recommended layout, where it predicts 1.4 mm.

   The folder holds the case templates, the Slurm job scripts with automatic stop detection and job chaining, and the post-processing that compares each run with the model. Its README explains how to run them.

## Folder layout

| Path | What it is |
|---|---|
| `paper/drain_stop.pdf` | The paper (compiled) |
| `paper/drain_stop.tex` | Paper source; `paper/figures/` (PDF + PNG of every figure) and `paper/generated/` (tables and numbers) are produced by the scripts |
| `drainstop/model.py` | Geometry, the USZ as-built hole map, layouts, hole and bore hydraulics, the quasi-static two-phase network solver, and the closed-form stop condition |
| `drainstop/transient.py` | Time-dependent simulation (the independent check) |
| `scripts/run_study.py` | All studies, written to `results/*.json` |
| `scripts/make_figures.py` | All figures, written to `paper/figures/` |
| `scripts/make_tables.py` | LaTeX tables and number macros, written to `paper/generated/` |
| `scripts/make_pictures.py` | Pictures and animation of the drain emptying (side view, 3D view, flow through every opening), written to `pictures/` |
| `pictures/` | The stills and the GIF animation (see "Pictures of the simulation") |
| `results/*.json` | Raw results of every study, one record per case. The figures and tables are built only from these files. |
| `results/recommended.json` | The chosen layout, read by the sensitivity, Monte Carlo, transient and mincap studies |
| `vof/` | 3D OpenFOAM check (interFoam) of two stop levels on ETH Euler: case generator, case templates, job scripts, analysis. See [`vof/README.md`](vof/README.md). |

## Rerun on another computer

The study needs Python 3.10 or newer with numpy and matplotlib; no scipy, gmsh or CFD software is needed. It was run with Python 3.12.10, numpy 2.5.3 and matplotlib 3.11.2.

```
cd drainstop
python -m pip install -r requirements.txt
```

**Reproduce everything.** The full run takes about 12 minutes on 12 cores. It uses all cores but two.

```
python scripts/run_study.py
python scripts/make_figures.py
python scripts/make_tables.py
python scripts/make_pictures.py
```

**Run single parts.** Each one rewrites its own `results/*.json`:

```
python scripts/run_study.py sweep transient mincap
```

On Windows, `py -3.12` can replace `python`.

**Rebuild the PDF.** Install [Tectonic](https://tectonic-typesetting.github.io/), a single executable that downloads the LaTeX packages it needs on first use, then run:

```
cd paper
tectonic -X compile drain_stop.tex
```

Any LaTeX distribution with `pdflatex` works too.

**Use the model directly:**

```python
# run from inside drainstop/
import sys; sys.path.insert(0, "scripts"); import run_study

m = run_study.make_model(dict(layout="custom", angle=30.0, d=1.0, n_x=18, family="custom", tilt=10))
print(m.analytic()["h_stop"], "mm  closed form")             # 1.35
print(m.descend(dh=0.1)["h_stop"], "mm  network simulation") # 1.36
```

**Change the design.** Edit `results/recommended.json`, for example the row angle, hole diameter `d` or holes per row `n_x`. Then rerun `sensitivity montecarlo transient mincap sizes_rec curves`, the figures, the tables and the PDF.

## Assumptions to keep in mind

- **Source.** −7 mmHg is held in the outlet tube by a wall regulator, modelled as ideal. The `pump` and `mincap` parts cover limited flow capacity.
- **Fluid.** The fluid is 50 wt% glycerol in water (ρ = 1126 kg/m³, μ = 6 mPa·s, γ = 0.068 N/m). Water and 40/80 % glycerol are in `FLUIDS` and in the sensitivity study.
- **Interior.** The drain interior is one open lumen. The as-built flutes and core change flow rates but not stop levels, since no liquid flows at the stop.
- **Simulation scope.** The model is reduced-order, not 3D volume-of-fluid CFD. Slug and churn flow in the bore, foaming and sloshing are only represented by the separated-phase bore model. The `vof/` cases are there to test this.

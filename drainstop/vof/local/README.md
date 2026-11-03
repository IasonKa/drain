# 3D CFD on a laptop or desktop: fixed-level OpenFOAM runs

This folder runs the 3D two-phase OpenFOAM model (interFoam, volume of fluid) on an ordinary Linux machine, or on Windows with the native OpenFOAM build, without the Euler cluster.

A full drainage run takes days. Each run here is instead a **snapshot at one pool level**:
- The pool starts at level h.
- The drain interior starts filled to the inner liquid level that the model predicts for that h, with air above it.
- About 0.12 s of flow is simulated. That is long enough for the air jets through the exposed openings and the liquid flow through the bore to settle, and short enough that the pool level hardly moves.

At that level the run gives:
- the liquid and air flow leaving through the outlet;
- the pressure inside the drain;
- the flow through every opening;
- the full pressure, velocity and liquid/air fields.

All of these are compared with the model (`drainstop/model.py`) at the same level.

| Case | Pool level | Model: liquid out | Model: air in | Model: pressure inside |
|---|---|---|---|---|
| `rec_tilt10:8`, recommended layout, outlet 10° up | 8 mm | 203 mL/min | 14.7 L/min | −427 Pa |
| `usz_tilt10:21`, USZ as built, outlet 10° up (the model stops it at 20.0 mm) | 21 mm | 148 mL/min | 23.3 L/min | −37.5 Pa |

## Run it on Linux (Ubuntu 22.04 / 24.04)

**1. OpenFOAM v2606** (openfoam.com), once:

```bash
curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install openfoam2606-default
```

**2. Python** with numpy and matplotlib, plus pyvista for the 3D view, once:

```bash
python3 -m venv ~/venvs/drain
~/venvs/drain/bin/pip install numpy matplotlib pyvista
```

**3. Run everything** from the repository. Use all cores, or set `NP`:

```bash
git clone https://github.com/IasonKa/drain.git && cd drain/drainstop
source /usr/lib/openfoam/openfoam2606/etc/bashrc
PY=~/venvs/drain/bin/python bash vof/local/run_all.sh
```

This builds the drain surfaces, then for each case writes it, meshes it (snappyHexMesh), runs interFoam in parallel and makes the pictures. They are copied to `vof/local/results/<case>/`.

**4. Share the results** by committing and pushing that folder:

```bash
git add vof/local/results && git commit -m "drainstop: CFD results" && git push
```

Other options:

```bash
NP=6 bash vof/local/run_all.sh rec_tilt10:8            # one case on 6 cores
bash vof/local/run_all.sh rec_tilt10:2 usz_tilt10:20   # other pool levels
AIR=10 bash vof/local/run_all.sh                       # 10x denser air, about 3x faster
FORCE=1 bash vof/local/run_all.sh                      # run finished cases again
```

**Air ×10.** Both resistances in series, the exposed openings and the bore, scale with the air density, so the vacuum in the drain and the stop level stay the same. The air velocities are √10 lower, though, so use real air for the velocity pictures.

## How long it takes

The recommended case has about 0.35 M cells and the USZ case about 0.67 M. Once the air jets form, the time step falls to about 5–10 µs, so 0.12 s takes 15–25 k steps. On one core a step takes about 3.4 s for the recommended case.

| Cores | Recommended | USZ |
|---|---|---|
| 8 (typical laptop) | 3–4 h | 6–8 h |
| 16 | about 2 h | about 4 h |

Meshing takes 2–7 minutes per case. Leave the laptop plugged in and stop it from sleeping.

## Files

| File | What it does |
|---|---|
| `run_all.sh` | Builds, meshes, runs and makes the pictures for a list of cases |
| `make_local_case.py` | Copies `vof/cases/<template>` to `runs/<case>/` and adapts it to a fixed level. It sets a lighter mesh, the initial liquid inside and outside the drain, a 0.12 s run, field output every 0.01 s, and sampled planes. It also stores the model's prediction (`model_state.json`). |
| `Allrun.local` | Becomes `runs/<case>/Allrun`. Runs blockMesh, surfaceFeatureExtract, snappyHexMesh, checkMesh, topoSet, setFields, decomposePar, interFoam and reconstructPar. |
| `render_cfd.py` | Makes the pictures and `summary.json` from the run's text output |
| `../env_windows.sh` | OpenFOAM environment for the native Windows build (Git Bash) |

**Pictures per case** (`results/<case>/`), averaged over the last 0.04 s:

| File | What it shows |
|---|---|
| `midplane.png` | Liquid/air, pressure and velocity magnitude in the plane through the drain axis, tilted as in the pool |
| `outlet_end.png` | Close-up of the outlet end with streamlines: the inner liquid level, the bore and the air jets |
| `row.png` | The plane through a row of openings |
| `sections.png` | Cross-sections through a submerged and an exposed opening, with velocity arrows |
| `openings.png` | Flow rate through every opening on the unrolled skin, CFD next to the model |
| `view3d.png` | 3D view: pressure on the drain, the liquid surface and the flow through every opening |
| `history.png` | Outlet liquid flow, outlet air flow and the pressure inside the drain against time, with the model values |
| `summary.json` | The averaged CFD numbers next to the model's |

## The mesh

The mesh has the same levels as the `coarse` Euler mesh (`vof/README.md`), except for the coarser drain interior and pool:

| Region | Cell size |
|---|---|
| Skin | 0.4 mm |
| Opening rims, tip cap, end face and bore | 0.2 mm, which puts 5 cells across a Ø1.0 mm opening |
| Drain interior away from the skin | 0.8 mm |
| Pool and air | 1.6 mm |

## On Windows instead

1. Install `OpenFOAM-v2606-windows-mingw.exe` from https://dl.openfoam.com/source/v2606/. It needs no admin rights: `/S /D=C:\Users\<you>\tools\OpenFOAM-v2606`. Set `OF_ROOT` in `vof/env_windows.sh`.
2. For parallel runs, also install MS-MPI from `thirdParty\msmpisetup.exe` (needs admin), then in the OpenFOAM `bin` folder run `cp libPstream.dll-msmpi libPstream.dll`. Without MS-MPI, run `cp libPstream.dll-dummy libPstream.dll` instead and use `NP=1`.
3. In Git Bash, run `PY="py -3.12" bash vof/local/run_all.sh`.

# Running drainsponge on ETH Euler

This folder holds everything needed to run the coupled OpenFOAM ⇄ FEBio sponge-tissue model on the ETH Euler cluster:

- one long, high-resolution run of one design (the main use), and
- optional smaller jobs: OpenFOAM-only, FEBio-only, and the design screening.

## 0. Rules for using Euler

Sources: https://docs.hpc.ethz.ch (user account, Slurm, storage, FAQ pages).

- **Who may use it.** Euler is for **members of ETH Zürich** with a valid ETH account. On first login you accept the cluster usage rules, which point to ETH's Acceptable Use Policy for telematics resources (BOT).
  - The account is personal. Do not share it, not even with a supervisor.
  - If this drain project is done together with a company, check with your ETH group or supervisor that running it on Euler is covered by the collaboration before you start.
- **Login nodes are for light work only:** editing, compiling, submitting. Everything else runs as Slurm jobs. Direct access to compute nodes is not allowed.
- **Request what you need.** Smaller requests start sooner.
  - Don't name partitions; Euler picks them.
  - After the test job, check the real usage with `myjobs -j <id>` and `get_inefficient_jobs`, then lower `--ntasks` / `--mem-per-cpu` if the request was too high.
- **Storage:**

  | Location | Size | Notes |
  |---|---|---|
  | `$HOME` | 50 GB | backed up |
  | `$SCRATCH` | 2.5 TB, 1 M files | **files older than ~2 weeks are deleted**, no backup |

  Runs go to `$SCRATCH`. Copy what you keep with `hpc/archive_results.sh`.
- **Self-resubmission is capped.** The run script resubmits itself only while the run is unfinished, at most `MAX_CHAIN` times (default 12). It never runs on its own forever.

## 1. One-time setup (≈ 1 h, mostly waiting for the FEBio build job)

```bash
ssh <ethz-user>@euler.ethz.ch                 # from the ETH network or VPN
git clone https://github.com/IasonKa/drain.git && cd drain/drainsponge
bash hpc/setup_euler.sh                       # Python venv + submits hpc/build_febio.sbatch
squeue -u $USER                               # wait for "febio-build" to finish
sbatch hpc/test_euler.sbatch                  # 15-30 min: modules, FEBio, OpenFOAM, restart chain
```

- **Modules** (`hpc/env_euler.sh`): `stack/2024-06 gcc/12.2.0 openmpi/4.1.6 openfoam/2306 python/3.11.6 cmake/3.27.7`. OpenFOAM v2306 is the ESI edition, which has `simpleFoam` and `fvOptions` as the cases need. If a module name has changed, run `module load stack/2024-06; module avail openfoam` and edit `env_euler.sh`.
- **FEBio** is not an Euler module. `build_febio.sbatch` builds **FEBio v4.13** from source (MIT licence), with the Pardiso solver (MKL from PyPI). It is the same version and build recipe as the drainfeb pilot run, and it installs into `$HOME/febio4`.
- **Check the test log** in `logs/drainsponge-test_<id>.out`. It must end with `exit code 0`, and `progress.log` must show `FEBio window 2 (restart)`. If it shows "falling back to rerun mode" instead, FEBio's restart did not work on this build. The run still works, but each window then re-runs from t = 0 (slower).

## 2. The main run: one design, all physics, long time

```bash
sbatch hpc/run_coupled.sbatch hpc/production_standoff.json
```

What `production_standoff.json` sets:

| Setting | Value |
|---|---|
| design | standoff ring (r 0.55–0.85 mm, 0.40 mm, 4 × 0.35 mm slots) |
| resolution | `fine` |
| fluid mesh | ~0.25 M cells |
| tissue mesh | ~0.22 M hex8 biphasic elements, ≈ 0.9 M unknowns |
| scenario | no free liquid (closed film edge): all drained liquid comes out of the tissue |
| tissue | E = 10 kPa, ν = 0.2, solid fraction 0.2, k = 1e-14 m⁴/(N·s), Holmes–Mow strain-dependent permeability (M = 4, α = 2), friction 0.2. **Placeholders.** |
| suction | 10 kPa (≈ −75 mmHg), switched on over 10 ms |
| time | **72 coupling windows**, log-spaced from 0.1 ms to **6 h**, each window 8 FEBio time steps (576 steps), up to 30 OpenFOAM ⇄ FEBio iterations per window |

Physics included:

- **Liquid:** OpenFOAM, full 3D Navier–Stokes (steady per window) in the channel, hole, slots and film.
- **Tissue:** FEBio biphasic, large deformation, transient.
- **Contact:** tissue–drain contact with friction on the skin, ring tops and walls, and hole wall.
- **Interface:** pore pressure continuity, film-pressure traction, seepage, and squeeze-film motion.

How the work is spread:

- **Inside a job (one node, 48 cores).** OpenFOAM runs with 48 MPI ranks (domain decomposition), then FEBio runs on the same 48 cores as OpenMP threads (Pardiso). They alternate, so no core sits idle.
- **Across jobs.** The run is cut into **time windows**. After every converged window the state is committed:
  - FEBio dump (`febio/committed/state_NNN.dmp`)
  - OpenFOAM solution
  - interface history (`interface_history.npz`, `state.json`)

  Each window starts from the previous one:
  - **FEBio** continues from the dump with a `febio_restart` file. That file redefines the interface-pressure load curves and adds one step, so the cost per window stays constant.
  - **OpenFOAM** restarts from the last solution on the moved mesh.
- **Chaining.** A job stops cleanly ~30 min before its 24 h limit and resubmits itself. The next job continues from the last committed window. You can also stop a run (`scancel`) at any time and restart it with the same `sbatch` line; at most the window that was in progress is lost.

Rough cost: about 1–3 h per window at `fine` with 48 cores, so **3–9 days of wall time in 24 h jobs**. These numbers are only estimates. Use the test job and the first windows (`progress.log` lists the seconds per window) to plan. For a quicker first look, run with `"resolution": "default"` (≈ 4× cheaper).

### Watching a run

```bash
tail -f $SCRATCH/drainsponge/production_standoff/progress.log
squeue -u $USER ; myjobs -j <jobid>
```

### Extending a finished run to a later time

The run keeps its state, so you can add windows after `t_end`:

```bash
EXTRA_ARGS="--extend-to-s 86400 --extend-windows 12" sbatch hpc/run_coupled.sbatch hpc/production_standoff.json
```

### Comparison runs (optional, same settings, independent jobs)

```bash
sbatch hpc/run_coupled.sbatch hpc/production_current.json        # current design (no ring)
sbatch hpc/run_coupled.sbatch hpc/production_standoff_pool.json  # free liquid available at the edge
```

## 3. Results

Everything is in `$SCRATCH/drainsponge/<run>/`:

| Path | Content |
|---|---|
| `progress.log` | every iteration: pressure residual, interface motion, timings |
| `state.json` | per window: film at the rim, min film, intrusion into the hole, seepage, iterations |
| `vtk/tissue.pvd`, `vtk/fluid.pvd`, `vtk/interface.pvd` | **open in ParaView and press play**: deformed tissue with pore pressure, liquid pressure/velocity, film thickness |
| `openfoam_windows/wNNN/` | converged OpenFOAM fields of every window |
| `replay/replay.xplt` | after `sbatch hpc/replay_febio.sbatch <run dir>`: the whole tissue history as one FEBio plot file for FEBio Studio |

Archive before scratch is purged:

```bash
bash hpc/archive_results.sh $SCRATCH/drainsponge/production_standoff     # -> $HOME/drainsponge_results/*.tar.gz
```

## 4. Smaller jobs

| Job | What it does | Command |
|---|---|---|
| OpenFOAM only | Darcy–Brinkman, rigid porous tissue | `sbatch hpc/run_porous.sbatch standoff closed` |
| FEBio only | tissue loaded with the preview's film pressures | `sbatch hpc/run_febio_oneway.sbatch standoff closed` |
| Ring screening | 46 designs, 1 core each, minutes | `sbatch --array=0-45%20 hpc/screen_standoff.sbatch`, then `sbatch --dependency=afterok:<id> hpc/screen_standoff.sbatch collect` |

## 5. If something fails

- **`simpleFoam not found`:** the module name or OpenFOAM environment changed. Edit `hpc/env_euler.sh`.
- **FEBio does not converge in a window:** the coupler stops with an error, and the last committed window is safe. Reduce the window size (more `n_windows`), or the FEBio step size (`febio_steps_per_window`), or the suction ramp speed, and resubmit. A resumed run keeps its saved `config.json`; change it there.
- **OpenFOAM does not reach the residual targets:** the hole is an orifice jet (Re up to ~1000 with free liquid). The coupler still takes the result after `end_iter` iterations. Check `openfoam/log.solver`.
- **Tissue reaches the channel (occlusion):** the coupler stops. That is a result, not a bug: the design failed for these tissue properties.

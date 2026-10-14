"""
Two-way OpenFOAM <-> FEBio coupling for the single-hole window, resumable across jobs.

Per coupling window [t_{n-1}, t_n] (log-spaced times), implicit sub-iterations k:

  1. FEBio (biphasic tissue) advances t_{n-1} -> t_n with the interface pressure history
     P_patch(t_1..t_{n-1}) (converged) and the current guess P_patch(t_n).
       restart mode : continue from the FEBio dump of window n-1 (constant cost per window)
       rerun mode   : run 0 -> t_n from scratch (fallback if restart fails)
     -> tissue-bottom positions at t_n and the Darcy flux leaving the tissue.
  2. The fluid mesh gets the new tissue bottom (vertical node moves only, same topology),
     tissue_iface gets the velocity  w_seep + d(bottom)/dt.
  3. OpenFOAM (steady, quasi-static liquid; MPI) -> pressure on tissue_iface -> patch means.
  4. Aitken-relaxed update of P_patch(t_n); converged when pressure and interface position
     stop changing (at least `min_iter` iterations).
After every converged window the state is committed (FEBio dump, OpenFOAM fields, interface
history, VTU snapshots), so a run can stop at any window and continue in the next job.

    python -m drainsponge.couple --out RUN [--config cfg.json] [--design standoff]
                                 [--resolution fine] [--walltime-s 86400] [--dry-run]
Exit codes: 0 finished, 3 stopped before the time limit (resubmit to continue), 1 error.
Needs `febio4` (cfg febio.exe) and a sourced OpenFOAM environment in the same shell.
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, subprocess, sys, time
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from . import config
from .geom import Window, WindowMesh
from .foam_mesh import FoamMesh
from .foam_case import write_case, latest_time, iface_pressure, vector_list, read_internal_scalar, read_internal_vector
from .febio_model import FebioModel, last_block_at

EXIT_DONE, EXIT_ERROR, EXIT_RESUBMIT = 0, 1, 3


def log(msg, path=None):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


class Coupler:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, os.path.abspath(out)
        os.makedirs(self.out, exist_ok=True)
        self.logf = os.path.join(self.out, "progress.log")
        self.w = Window.from_dict(cfg["window"])
        closed = cfg.get("boundary", {}).get("film_edge", "open") == "closed"
        self.fl = WindowMesh(self.w, zones=("channel", "hole", "stand", "film"))
        self.fm = FoamMesh(self.fl, closed_edge=closed)
        self.tis = WindowMesh(self.w, zones=("tissue",))
        self.fe = FebioModel(cfg, self.tis, self.fl)
        self.case = os.path.join(self.out, "openfoam")
        T = cfg["time"]
        n = T["n_windows"]
        self.times = T["t_first_s"] * (T["t_end_s"] / T["t_first_s"]) ** (np.arange(n) / (n - 1))
        # continue a finished run for longer: extra log-spaced windows after t_end
        if T.get("extend_to_s", 0) > T["t_end_s"] and T.get("extend_windows", 0) > 0:
            extra = np.geomspace(T["t_end_s"], T["extend_to_s"], int(T["extend_windows"]) + 1)[1:]
            self.times = np.r_[self.times, extra]
        self.of_faces = self.fm.patch_faces("tissue_iface")
        self.of_quad = self.fl.cell_plan[self.fm.owner[self.of_faces]]
        self.state_file = os.path.join(self.out, "state.json")
        self.hist_file = os.path.join(self.out, "interface_history.npz")
        if os.path.exists(self.state_file):                      # resume
            self.state = json.load(open(self.state_file))
            H = np.load(self.hist_file)
            self.P = [p for p in H["P"]]
            self.zt = [z for z in H["zt"]]
            self.fl.set_tissue_bottom(self.zt[-1])
            log(f"resuming after window {self.state['n_done']} / {len(self.times)} "
                f"(febio mode: {self.state['mode']})", self.logf)
        else:
            write_case(self.fm, self.case, cfg, kind="fluid")
            json.dump(cfg, open(os.path.join(self.out, "config.json"), "w"), indent=1)
            self.state = dict(n_done=0, mode="restart" if cfg["coupling"].get("febio_restart", True) else "rerun",
                              log=[], window_seconds=[])
            self.P = [np.zeros(self.fe.K)]                       # converged patch pressure (Pa) at t_0..t_n
            self.zt = [self.fl.zt.copy()]                        # converged tissue-bottom height field
            self._save()

    # ------------------------------------------------------------------ state
    def _save(self):
        tmp = self.state_file + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        np.savez(self.hist_file, P=np.array(self.P), zt=np.array(self.zt), plan=self.fl.plan,
                 times=self.times, patch_node=self.fe.patch_node, patch_quad=self.fe.patch_quad)
        os.replace(tmp, self.state_file)

    def committed_dump(self, n):
        return os.path.join(self.out, "febio", "committed", f"state_{n:03d}.dmp")

    # ------------------------------------------------------------------ FEBio
    def _febio_env(self):
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", str(self.cfg["febio"].get("threads", 4)))
        return env

    def run_febio(self, n, P_new, wd):
        """Advance the tissue to t_n. Returns (zt_n on the plan grid, seepage per plan quad [mm/s], run dir)."""
        if os.path.isdir(wd):
            shutil.rmtree(wd)
        os.makedirs(wd)
        hist = np.array(self.P[:n] + [P_new])
        exe = self.cfg["febio"]["exe"]
        exe = list(exe) if isinstance(exe, (list, tuple)) else [exe]
        use_restart = self.state["mode"] == "restart" and n > 1 and os.path.exists(self.committed_dump(n - 1))
        if use_restart:
            self.fe.write_restart(os.path.join(wd, "restart.feb"), self.committed_dump(n - 1), self.times[:n], hist)
            cmd = exe + ["-r", "restart.feb", "-dump=state.dmp", "-silent"]
        else:
            self.fe.write(os.path.join(wd, "tissue.feb"), self.times[:n], hist,
                          plot="PLOT_NEVER" if self.state["mode"] == "restart" else "PLOT_FINAL")
            cmd = exe + ["-i", "tissue.feb", "-dump=state.dmp", "-silent"]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=wd, capture_output=True, text=True, env=self._febio_env())
        open(os.path.join(wd, "febio_stdout.txt"), "w").write(r.stdout + "\n" + r.stderr)
        ok = r.returncode == 0 or any("N O R M A L   T E R M I N A T I O N" in open(f, errors="ignore").read()
                                      for f in glob.glob(os.path.join(wd, "*.log")))
        t_n = self.times[n - 1]
        logs = self._find_logs(wd, n)
        if not ok or logs is None:
            raise RuntimeError(f"FEBio failed in {wd} (rc={r.returncode}, logs={'found' if logs else 'missing'})")
        log(f"   FEBio window {n} ({'restart' if use_restart else 'from t=0'}) {time.time()-t0:.0f}s", self.logf)
        xyz_file, flux_file = logs
        _, ids, v = last_block_at(xyz_file, t_n)
        z1 = self._height_field(ids, v)
        _, eids, W = last_block_at(flux_file, t_n)
        wmap = dict(zip(eids.tolist(), W))
        w_face = np.array([wmap.get(e + 1, [0, 0, 0]) for e in self.fe.bottom_elem])    # mm/s per bottom face
        wq = np.zeros((len(self.fl.quads), 3))
        wq[self.fe.bottom_quad] = w_face
        return z1, wq, wd

    def _find_logs(self, wd, n):
        """FEBio writes its logs next to the input or in the working directory; after a restart
        they may be appended to the files of the original run. Take the newest pair with data at t_n."""
        t_n = self.times[n - 1]
        cands = [wd] + sorted(glob.glob(os.path.join(self.out, "febio", "w*", "it*")))
        best = None
        for d in cands:
            a, b = os.path.join(d, "bottom_xyz.txt"), os.path.join(d, "bottom_flux.txt")
            if os.path.exists(a) and os.path.exists(b):
                try:
                    tb, _, _ = last_block_at(a, t_n)
                except Exception:
                    continue
                if abs(tb - t_n) <= 1e-6 * max(1.0, t_n):
                    m = os.path.getmtime(a)
                    if best is None or m > best[0]:
                        best = (m, a, b)
        return (best[1], best[2]) if best else None

    def _height_field(self, ids, v):
        """Deformed tissue-bottom node positions -> height at the fixed plan grid points."""
        pts = v[:, :2]
        f = LinearNDInterpolator(pts, v[:, 2])
        g = NearestNDInterpolator(pts, v[:, 2])
        z = f(self.fl.plan)
        bad = np.isnan(z)
        z[bad] = g(self.fl.plan[bad])
        return z

    def _find_dump(self, wd):
        cands = glob.glob(os.path.join(wd, "*.dmp"))
        return max(cands, key=os.path.getmtime) if cands else None

    # ------------------------------------------------------------------ OpenFOAM
    def _set_outlet_pressure(self, t):
        """Suction switched on over suction.ramp_s (same ramp as the preview model)."""
        cfg = self.cfg
        ramp = min(1.0, t / max(cfg["suction"].get("ramp_s", 0.0), 1e-30))
        p_out = -cfg["suction"]["dp_Pa"] * ramp / cfg["fluid"]["rho_kg_m3"]
        ppath = os.path.join(self.case, "0", "p")
        if os.path.exists(ppath + ".gz"):
            import gzip
            txt = gzip.open(ppath + ".gz", "rt").read()
            os.remove(ppath + ".gz")
        else:
            txt = open(ppath).read()
        block = ("outlet\n    {\n        type            fixedValue;\n"
                 f"        value           uniform {p_out:.8g};\n    }}")
        txt, nsub = re.subn(r"outlet\s*\{[^{}]*\}", lambda m: block, txt, count=1)
        if nsub != 1:
            raise RuntimeError("could not find outlet in 0/p")
        open(ppath, "w", newline="\n").write(txt)

    def run_openfoam(self, zt, U_iface, t=None):
        self.fl.set_tissue_bottom(zt)
        rep, _ = self.fm.check()
        if not rep["ok"]:
            raise RuntimeError(f"fluid mesh invalid after moving the tissue: {rep['problems']}")
        if self.fl.occluded:
            raise RuntimeError("tissue reaches the channel (occlusion) - beyond this model")
        self.fm.write_points(self.case)
        lt = latest_time(self.case)                               # restart from the last solution
        if lt:
            for f in os.listdir(os.path.join(self.case, lt)):
                base = f.split(".")[0]
                if base not in ("U", "p"):
                    continue
                for old in (base, base + ".gz"):
                    if os.path.exists(os.path.join(self.case, "0", old)):
                        os.remove(os.path.join(self.case, "0", old))
                shutil.copy(os.path.join(self.case, lt, f), os.path.join(self.case, "0", f))
            for d in os.listdir(self.case):
                try:
                    if float(d) > 0:
                        shutil.rmtree(os.path.join(self.case, d))
                except ValueError:
                    pass
        upath = os.path.join(self.case, "0", "U")
        if os.path.exists(upath + ".gz"):
            import gzip
            txt = gzip.open(upath + ".gz", "rt").read()
            os.remove(upath + ".gz")
        else:
            txt = open(upath).read()
        block = ("tissue_iface\n    {\n        type            fixedValue;\n        value           "
                 + vector_list(U_iface) + "        ;\n    }")
        txt, nsub = re.subn(r"tissue_iface\s*\{[^{}]*\}", lambda m: block, txt, count=1)
        if nsub != 1:
            raise RuntimeError("could not find tissue_iface in 0/U")
        open(upath, "w", newline="\n").write(txt)
        if t is not None:
            self._set_outlet_pressure(t)
        env = dict(os.environ, NP=str(self.cfg["openfoam"]["n_procs"]), SKIP_VTK="1")
        t0 = time.time()
        r = subprocess.run(["bash", "Allrun"], cwd=self.case, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("OpenFOAM failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        log(f"   OpenFOAM {time.time()-t0:.0f}s", self.logf)
        p_face = iface_pressure(self.fm, self.case, self.cfg["fluid"]["rho_kg_m3"])
        pq = np.zeros(len(self.fl.quads))
        pq[self.of_quad] = p_face
        return pq

    # ------------------------------------------------------------------ commit + outputs
    def commit(self, n, P, z1, wd, pq, wq, iters, secs):
        dmp = self._find_dump(wd)
        if dmp:
            os.makedirs(os.path.dirname(self.committed_dump(n)), exist_ok=True)
            shutil.move(dmp, self.committed_dump(n))
            old = self.committed_dump(n - 2)                    # keep the last two dumps only
            if n > 2 and os.path.exists(old):
                os.remove(old)
        elif self.state["mode"] == "restart":
            log("   no FEBio dump found -> switching to rerun mode", self.logf)
            self.state["mode"] = "rerun"
        self.P.append(P)
        self.zt.append(z1)
        h = self.fl.film_thickness()
        rec = dict(window=n, t=float(self.times[n - 1]), iters=iters, seconds=secs,
                   p_iface_min_Pa=float(P.min()), p_iface_mean_Pa=float(np.average(P)),
                   film_rim_um=float(h[self.fl.node_rim].mean() * 1e3),
                   min_film_um=float(h[~self.fl.node_hole].min() * 1e3),
                   intrusion_um=float(self.fl.intrusion * 1e3),
                   seep_out_mm3_s=float(-np.sum(wq[:, 2] * self.fe.quad_area)))
        self.state["log"].append(rec)
        self.state["window_seconds"].append(secs)
        self.state["n_done"] = n
        self._save()
        if self.cfg["coupling"].get("save_every_window", True):
            try:
                from .post import write_window_snapshot
                write_window_snapshot(self, n, wd, pq)
            except Exception as e:                               # outputs must never stop the run
                log(f"   snapshot skipped: {e}", self.logf)
        # keep a copy of the converged OpenFOAM fields
        lt = latest_time(self.case)
        if lt:
            dst = os.path.join(self.out, "openfoam_windows", f"w{n:03d}")
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(os.path.join(self.case, lt)):
                if f.split(".")[0] in ("U", "p", "wallShearStress"):
                    shutil.copy(os.path.join(self.case, lt, f), dst)
        log(f"window {n}/{len(self.times)} t={self.times[n-1]:.4g}s converged in {iters} it, {secs:.0f}s; "
            f"p_min={P.min():.0f}Pa film@rim={rec['film_rim_um']:.1f}um intrusion={rec['intrusion_um']:.1f}um", self.logf)

    # ------------------------------------------------------------------ main loop
    def run(self, deadline=None, dry_run=False):
        C = self.cfg["coupling"]
        dp = self.cfg["suction"]["dp_Pa"]
        N = len(self.times)
        for n in range(self.state["n_done"] + 1, N + 1):
            if deadline is not None:
                ws = self.state["window_seconds"]
                need = 1.5 * max(ws[-3:]) if ws else 0.0
                if time.time() + need > deadline:
                    log(f"stopping before window {n}: not enough wall time left (need ~{need:.0f}s)", self.logf)
                    return EXIT_RESUBMIT
            t_start = time.time()
            # initial guess: linear extrapolation in log time of the last two windows
            P = self.P[-1].copy()
            if len(self.P) > 2:
                P = self.P[-1] + 0.5 * (self.P[-1] - self.P[-2])
            r_old, omega = None, C["omega0"]
            z_last = self.zt[-1]
            for k in range(C["max_iter"]):
                wd = os.path.join(self.out, "febio", f"w{n:03d}", f"it{k:02d}")
                if dry_run:
                    hist = np.array(self.P[:n] + [P])
                    os.makedirs(wd, exist_ok=True)
                    self.fe.write(os.path.join(wd, "tissue.feb"), self.times[:n], hist)
                    self.fe.write_restart(os.path.join(wd, "restart_example.feb"), self.committed_dump(max(n - 1, 1)),
                                          self.times[:n], hist)
                    log(f"dry run: wrote {wd} and {self.case}", self.logf)
                    return EXIT_DONE
                try:
                    z1, wq, wdir = self.run_febio(n, P, wd)
                except RuntimeError as e:
                    if self.state["mode"] == "restart" and n > 1:
                        log(f"   FEBio restart failed ({e}); falling back to rerun mode", self.logf)
                        self.state["mode"] = "rerun"
                        self._save()
                        z1, wq, wdir = self.run_febio(n, P, wd)
                    else:
                        raise
                dt = self.times[n - 1] - (self.times[n - 2] if n > 1 else 0.0)
                vz = (z1 - self.zt[-1]) / dt                               # mm/s, plan nodes
                vq = np.mean(vz[self.fl.quads], axis=1)
                Uq = wq.copy()
                Uq[:, 2] += vq
                U = Uq[self.of_quad] * 1e-3                                 # m/s on tissue_iface faces
                pq = self.run_openfoam(z1, U, t=self.times[n - 1])
                P_new = self.fe.patch_average(pq)
                r = P_new - P
                if r_old is not None:                                       # Aitken
                    dr = r - r_old
                    omega = -omega * float(r_old @ dr) / max(float(dr @ dr), 1e-30)
                    omega = float(np.clip(omega, 0.05, 1.0))
                P = P + omega * r
                r_old = r
                dz = float(np.max(np.abs(z1 - z_last)))
                z_last = z1
                res = float(np.max(np.abs(r)) / dp)
                log(f" window {n} t={self.times[n-1]:.4g}s iter {k}: |dP|/dp={res:.2e} dz={dz*1e3:.3f}um omega={omega:.2f}",
                    self.logf)
                if k + 1 >= C.get("min_iter", 2) and res < C["tol_p_rel"] and dz < C["tol_disp_mm"]:
                    break
            else:
                log(f" window {n}: not converged after {C['max_iter']} iterations - continuing with the last state",
                    self.logf)
            self.commit(n, P, z1, wdir, pq, wq, k + 1, time.time() - t_start)
            # sub-iteration directories are only needed until the window is committed
            # (window 1 is kept whole: FEBio may append restart logs to its directory)
            if n > 1:
                for d in glob.glob(os.path.join(self.out, "febio", f"w{n:03d}", "it*")):
                    if os.path.abspath(d) != os.path.abspath(wdir):
                        shutil.rmtree(d, ignore_errors=True)
        self.state["finished"] = True
        self.state["n_windows"] = len(self.times)
        self._save()
        log("all windows done", self.logf)
        return EXIT_DONE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--design", default=None, choices=list(config.DESIGNS))
    ap.add_argument("--resolution", default=None, choices=list(config.RESOLUTIONS))
    ap.add_argument("--out", default="runs/coupled")
    ap.add_argument("--walltime-s", type=float, default=None, help="stop cleanly before this much time has passed")
    ap.add_argument("--margin-s", type=float, default=1800, help="safety margin before the wall time")
    ap.add_argument("--np", type=int, default=None, help="OpenFOAM MPI ranks (default: config or $SLURM_NTASKS)")
    ap.add_argument("--threads", type=int, default=None, help="FEBio OpenMP threads (default: config or $SLURM_NTASKS)")
    ap.add_argument("--febio-exe", default=None, help="FEBio binary (default: config febio.exe)")
    ap.add_argument("--extend-to-s", type=float, default=None, help="continue an existing run up to this time")
    ap.add_argument("--extend-windows", type=int, default=12, help="number of extra windows for --extend-to-s")
    ap.add_argument("--dry-run", action="store_true", help="only write the first FEBio + OpenFOAM inputs")
    a = ap.parse_args()
    t_begin = time.time()
    saved = os.path.join(a.out, "config.json")
    if os.path.exists(saved):                       # a resumed run always uses the configuration it started with
        cfg = json.load(open(saved))
        if a.extend_to_s:                           # ... except for extending it in time
            cfg["time"]["extend_to_s"] = a.extend_to_s
            cfg["time"]["extend_windows"] = a.extend_windows
            json.dump(cfg, open(saved, "w"), indent=1)
    else:
        cfg = config.load(a.config, design=a.design, resolution=a.resolution)
    ntask = os.environ.get("SLURM_NTASKS")
    cfg["openfoam"]["n_procs"] = a.np or (int(ntask) if ntask else cfg["openfoam"]["n_procs"])
    cfg["febio"]["threads"] = a.threads or (int(ntask) if ntask else cfg["febio"]["threads"])
    os.environ["OMP_NUM_THREADS"] = str(cfg["febio"]["threads"])
    if a.febio_exe:
        cfg["febio"]["exe"] = a.febio_exe
    deadline = t_begin + a.walltime_s - a.margin_s if a.walltime_s else None
    try:
        code = Coupler(cfg, a.out).run(deadline=deadline, dry_run=a.dry_run)
    except Exception as e:
        log(f"ERROR: {e}", os.path.join(os.path.abspath(a.out), "progress.log"))
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()

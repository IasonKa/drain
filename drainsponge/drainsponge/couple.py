"""
Partitioned OpenFOAM <-> FEBio coupling for the single-hole window.

Per coupling window [t_{n-1}, t_n] (log-spaced times), implicit sub-iterations:

  1. FEBio (biphasic tissue) runs 0 -> t_n with the interface pressure history
     P(x, t_1..t_{n-1}) (converged) and the current guess P(x, t_n).
     -> tissue-bottom positions at t_{n-1}, t_n and Darcy flux leaving the tissue.
  2. The fluid mesh gets the new tissue bottom (vertical node moves only, same
     topology), and tissue_iface gets the velocity  w_seep + d(bottom)/dt.
  3. OpenFOAM (steady, quasi-static film) -> pressure on tissue_iface -> P_new(x, t_n).
  4. Aitken-relaxed update of P(x, t_n); stop when the pressure and the interface
     position stop changing.

Interface nodes are shared 1:1 (same plan grid), only the deformed in-plane position
of the FEBio nodes needs an interpolation back to the fixed plan grid.
FEBio is re-run from t = 0 each time (simple and robust; the tissue model is small).

    python -m drainsponge.couple --out runs/coupled [--config cfg.json] [--dry-run]
Needs `febio4` on PATH (or febio.exe in config) and a sourced OpenFOAM environment.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from . import config
from .geom import Window, WindowMesh
from .foam_mesh import FoamMesh
from .foam_case import write_case, latest_time, iface_pressure, vector_list
from .febio_model import FebioModel, read_log_blocks


class Coupler:
    def __init__(self, cfg, out):
        self.cfg, self.out = cfg, out
        os.makedirs(out, exist_ok=True)
        self.w = Window.from_dict(cfg["window"])
        self.fl = WindowMesh(self.w, zones=("channel", "hole", "film"))
        self.fm = FoamMesh(self.fl)
        self.tis = WindowMesh(self.w, zones=("tissue",))
        self.fe = FebioModel(cfg, self.tis, self.fl)
        self.case = os.path.join(out, "openfoam")
        write_case(self.fm, self.case, cfg, kind="fluid")
        T = cfg["time"]
        n = T["n_windows"]
        self.times = T["t_first_s"] * (T["t_end_s"] / T["t_first_s"]) ** (np.arange(n) / (n - 1))
        npl = len(self.fl.plan)
        self.P = [np.zeros(npl)]                         # converged interface pressure (Pa), plan order
        self.zt = [self.fl.zt.copy()]                    # converged tissue-bottom height field
        # face <-> plan quad maps
        self.of_faces = self.fm.patch_faces("tissue_iface")
        self.of_quad = self.fl.cell_plan[self.fm.owner[self.of_faces]]
        b = ~self.tis.f_internal & (self.tis.f_patch == "tissue_bottom")
        self.fe_face_quad = self.tis.cell_plan[self.tis.f_owner[np.nonzero(b)[0]]]
        self.fe_bottom_plan = self.tis.node_plan[self.fe.bottom_set]
        self.log = []

    # ------------------------------------------------------------------ helpers
    def _quad_to_nodes(self, vq, quad_ids):
        """Face values (per plan quad) -> plan nodes (average of the quads around each node)."""
        acc = np.zeros(len(self.fl.plan)); cnt = np.zeros(len(self.fl.plan))
        for k in range(4):
            nd = self.fl.quads[quad_ids, k]
            np.add.at(acc, nd, vq); np.add.at(cnt, nd, 1)
        return acc / np.maximum(cnt, 1)

    def run_febio(self, n, P_new, workdir):
        os.makedirs(workdir, exist_ok=True)
        feb = os.path.join(workdir, "tissue.feb")
        hist = np.array(self.P[:n] + [P_new])
        self.fe.write(feb, self.times[:n], hist)
        exe = self.cfg["febio"]["exe"]
        r = subprocess.run([exe, "-i", "tissue.feb", "-silent"], cwd=workdir, capture_output=True, text=True)
        logtxt = open(os.path.join(workdir, "tissue.log")).read() if os.path.exists(os.path.join(workdir, "tissue.log")) else r.stdout
        if "N O R M A L   T E R M I N A T I O N" not in logtxt:
            raise RuntimeError(f"FEBio did not finish normally in {workdir}")
        return self.read_febio(workdir, n)

    def read_febio(self, workdir, n):
        """Tissue bottom height field at t_{n-1}, t_n (plan grid) and seepage per plan quad (mm/s)."""
        xyz = read_log_blocks(os.path.join(workdir, "bottom_xyz.txt"))
        flx = read_log_blocks(os.path.join(workdir, "bottom_flux.txt"))
        tl = np.array([b[0] for b in xyz])
        t_now = self.times[n - 1]
        t_prev = self.times[n - 2] if n > 1 else 0.0

        def height(t):
            if t <= 0:
                return self.zt[0]
            _, ids, v = xyz[int(np.argmin(np.abs(tl - t)))]
            plan = self.tis.node_plan[ids - 1]
            pts = v[:, :2]
            f = LinearNDInterpolator(pts, v[:, 2]); g = NearestNDInterpolator(pts, v[:, 2])
            z = f(self.fl.plan); bad = np.isnan(z); z[bad] = g(self.fl.plan[bad])
            return z
        z1, z0 = height(t_now), height(t_prev)
        _, eids, W = flx[int(np.argmin(np.abs(np.array([b[0] for b in flx]) - t_now)))]
        wmap = dict(zip(eids.tolist(), W))
        w_face = np.array([wmap.get(e + 1, [0, 0, 0]) for e in self.fe.bottom_elem])   # mm/s, per bottom face
        return z0, z1, w_face, t_now - t_prev

    def run_openfoam(self, zt, U_iface):
        self.fl.set_tissue_bottom(zt)
        rep, _ = self.fm.check()
        if not rep["ok"]:
            raise RuntimeError(f"fluid mesh invalid after moving the tissue: {rep['problems']}")
        if self.fl.occluded:
            raise RuntimeError("tissue reaches the channel (occlusion) - beyond this model")
        self.fm.write_points(self.case)
        # restart from the last solution: latest time -> 0, then set the interface velocity
        lt = latest_time(self.case)
        if lt:
            for f in ("U", "p"):
                shutil.copy(os.path.join(self.case, lt, f), os.path.join(self.case, "0", f))
            for d in os.listdir(self.case):
                try:
                    if float(d) > 0:
                        shutil.rmtree(os.path.join(self.case, d))
                except ValueError:
                    pass
        upath = os.path.join(self.case, "0", "U")
        txt = open(upath).read()
        block = ("tissue_iface\n    {\n        type            fixedValue;\n        value           "
                 + vector_list(U_iface) + "        ;\n    }")
        txt, nsub = re.subn(r"tissue_iface\s*\{[^{}]*\}", lambda m: block, txt, count=1)
        if nsub != 1:
            raise RuntimeError("could not find tissue_iface in 0/U")
        open(upath, "w", newline="\n").write(txt)
        env = dict(os.environ, NP=str(self.cfg["openfoam"]["n_procs"]))
        r = subprocess.run(["bash", "Allrun"], cwd=self.case, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("OpenFOAM failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
        p_face = iface_pressure(self.fm, self.case, self.cfg["fluid"]["rho_kg_m3"])
        return self._quad_to_nodes(p_face, self.of_quad)

    # ------------------------------------------------------------------ main loop
    def run(self, dry_run=False):
        C = self.cfg["coupling"]
        dp = self.cfg["suction"]["dp_Pa"]
        for n in range(1, len(self.times) + 1):
            P = self.P[-1].copy()
            r_old, omega = None, C["omega0"]
            for k in range(C["max_iter"]):
                wd = os.path.join(self.out, "febio", f"w{n:03d}")
                if dry_run:
                    self.fe.write(os.path.join(wd, "tissue.feb"), self.times[:n], np.array(self.P[:n] + [P]))
                    print(f"dry run: wrote {wd}/tissue.feb and {self.case}")
                    return
                z0, z1, w_face, dt = self.run_febio(n, P, wd)
                vz = (z1 - z0) / dt                                    # mm/s, plan nodes
                vq = np.mean(vz[self.fl.quads], axis=1)                # per plan quad
                wq = np.zeros((len(self.fl.quads), 3))
                wq[self.fe_face_quad] = w_face
                U = (wq[self.of_quad] + np.c_[np.zeros((len(vq), 2)), vq][self.of_quad]) * 1e-3   # m/s
                P_new = self.run_openfoam(z1, U)
                r = P_new - P
                if r_old is not None:                                  # Aitken
                    dr = r - r_old
                    omega = -omega * float(r_old @ dr) / max(float(dr @ dr), 1e-30)
                    omega = float(np.clip(omega, 0.05, 1.0))
                P = P + omega * r
                r_old = r
                dz = float(np.max(np.abs(z1 - self.zt[-1]))) if k == 0 else float(np.max(np.abs(z1 - z_last)))
                z_last = z1
                res = float(np.max(np.abs(r)) / dp)
                print(f"window {n}/{len(self.times)} t={self.times[n-1]:.4g}s iter {k}: |dP|/dp={res:.2e} "
                      f"dz={dz*1e3:.3f}um omega={omega:.2f}")
                if res < C["tol_p_rel"] and dz < C["tol_disp_mm"]:
                    break
            self.P.append(P); self.zt.append(z1)
            h = self.fl.film_thickness()
            self.log.append(dict(t=float(self.times[n - 1]), iters=k + 1, p_min_Pa=float(P.min()),
                                 film_rim_um=float(h[self.fl.node_rim].mean() * 1e3),
                                 intrusion_um=float(self.fl.intrusion * 1e3)))
            json.dump(dict(log=self.log, times=self.times.tolist()), open(os.path.join(self.out, "coupling_log.json"), "w"), indent=1)
            np.savez(os.path.join(self.out, "interface_history.npz"), P=np.array(self.P), zt=np.array(self.zt),
                     plan=self.fl.plan, times=self.times[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/coupled")
    ap.add_argument("--dry-run", action="store_true", help="only write the first FEBio + OpenFOAM inputs")
    a = ap.parse_args()
    cfg = config.load(a.config)
    Coupler(cfg, a.out).run(dry_run=a.dry_run)


if __name__ == "__main__":
    main()

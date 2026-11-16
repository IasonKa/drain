"""Flushing score of a solved blocked-sheath case: how fast the old fluid in the lumen is replaced.

  python flush_score.py runs/<case> [--n 3000] [--T 600] [--D 1e-9]   ->  results/<case>/flush.json

The lumen is filled with "old fluid" markers (uniform by volume). Each marker is carried by the steady flow
(midpoint rule, its own time step from the local cell size and speed) and also diffuses (random walk with
diffusivity D, m2/s; 1e-9 is a small dissolved molecule in water). A marker is gone when it passes into the
suction tube. Without the diffusion, markers inside a closed vortex or a groove would never leave.
"""
import argparse, json, os, time
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
YA, YB = -59.5, 59.9
Y_EXIT = YA - 12.0                                           # inside the suction tube
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--T", type=float, default=600.0)
ap.add_argument("--D", type=float, default=1e-9)
ap.add_argument("--cfl", type=float, default=0.35)
ap.add_argument("--dtmax", type=float, default=0.5)
a = ap.parse_args()
name = os.path.basename(os.path.normpath(a.run))
out = os.path.join(HERE, "results", name)
os.makedirs(out, exist_ok=True)
case = json.load(open(os.path.join(a.run, "case.json")))
s_mm = case.get("s_mm", 0.0)
m = np.load(os.path.join(a.run, "mesh.npz"))
r = np.load(os.path.join(a.run, "result.npz"))
xyz, tets = m["xyz"].astype(float), m["tets"]
g = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                        np.full(len(tets), pv.CellType.TETRA), xyz)
vn = r["v"].astype(float) * 1e3                               # mm/s
X = xyz[tets]
E = np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1))
vol = np.abs(np.linalg.det(E)) / 6
Minv = np.linalg.inv(E)
hloc = (8.5 * vol) ** (1 / 3)                                 # edge length of a regular tet of that volume
D = a.D * 1e6                                                 # mm2/s
rng = np.random.default_rng(0)

cy = X[:, :, 1].mean(1)
vol_l = np.where((cy > YA) & (cy < YB), vol, 0.0)
pick = rng.choice(len(tets), a.n, p=vol_l / vol_l.sum())
P = np.einsum("ni,nij->nj", rng.dirichlet(np.ones(4), a.n), X[pick])
seeds = P.copy()
cell = pick.copy()
tcur = np.zeros(a.n)
t_exit = np.full(a.n, np.inf)
act = np.ones(a.n, bool)


def vel(c, p):
    lam = np.einsum("nij,nj->ni", Minv[c], p - X[c, 0])
    return np.einsum("ni,nij->nj", np.c_[1 - lam.sum(1), lam], vn[tets[c]])


t0, it = time.time(), 0
while act.any() and it < 400000:
    i = np.flatnonzero(act)
    p, c = P[i], cell[i]
    v1 = vel(c, p)
    dt = np.minimum(a.cfl * hloc[c] / np.maximum(np.linalg.norm(v1, axis=1), 1e-9), a.dtmax)
    dt = np.minimum(dt, a.T - tcur[i] + 1e-9)
    mid = p + 0.5 * dt[:, None] * v1
    cm = np.asarray(g.find_containing_cell(mid))
    v2 = v1.copy()
    ok = cm >= 0
    v2[ok] = vel(cm[ok], mid[ok])
    pn = p + dt[:, None] * v2 + np.sqrt(2 * D * dt)[:, None] * rng.normal(size=p.shape)
    cn = np.asarray(g.find_containing_cell(pn))
    gone = pn[:, 1] < Y_EXIT
    stay = (cn < 0) & ~gone                                   # stepped through a wall: do not move this time
    pn[stay], cn[stay] = p[stay], c[stay]
    P[i], cell[i] = pn, np.where(cn >= 0, cn, c)
    tcur[i] += dt
    t_exit[i[gone]] = tcur[i[gone]]
    act[i[gone]] = False
    act[i[~gone & (tcur[i] >= a.T)]] = False
    it += 1
    if it % 5000 == 0:
        print(f"iteration {it}: {int(act.sum())} markers still moving, slowest clock {tcur[act].min() if act.any() else a.T:.0f} s, "
              f"wall {time.time() - t0:.0f} s", flush=True)

tt = np.linspace(0, a.T, 121)
left = np.array([(t_exit > t).mean() for t in tt])
behind = seeds[:, 1] > YB - s_mm                              # between the tip and the irrigation outlet


def frac(mask, t):
    return float((t_exit[mask] > t).mean()) if mask.any() else None


res = dict(case=name, s_mm=s_mm, d_mm=case.get("d_mm", 1.0), n=a.n, T=a.T, D_m2_s=a.D, iterations=it,
           old_left={f"{int(t)}s": float((t_exit > t).mean()) for t in (60, 120, 300, 600) if t <= a.T},
           volume_behind_outlet=float(behind.mean()),
           old_left_behind_outlet={f"{int(t)}s": frac(behind, t) for t in (300, 600) if t <= a.T},
           old_left_ahead_of_outlet={f"{int(t)}s": frac(~behind, t) for t in (300, 600) if t <= a.T},
           curve=dict(t=tt.tolist(), left=left.tolist()))
json.dump(res, open(os.path.join(out, "flush.json"), "w"), indent=1)
np.savez(os.path.join(out, "flush_markers.npz"), seeds=seeds.astype(np.float32), t_exit=t_exit, end=P.astype(np.float32))
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))

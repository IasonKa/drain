"""
Pictures and numbers for one flow case (after run_flow.py).
  python render_flow.py runs/<case>
Writes into results/<case>/: planes.png, sections.png, streamlines.png, flushing.png, summary.json
"""
import json, os, sys
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
XC, ZC = 0.78, -4.82
YA, YB = -59.5, 59.9
run = sys.argv[1]
name = os.path.basename(os.path.normpath(run))
out = os.path.join(HERE, "results", name)
os.makedirs(out, exist_ok=True)
case = json.load(open(os.path.join(run, "case.json")))
m = np.load(os.path.join(run, "mesh.npz"))
r = np.load(os.path.join(run, "result.npz"))
xyz, tets, tris, lab, prof = m["xyz"], m["tets"], m["tris"], m["lab"], m["profile"]
v = r["v"] * 1e3                                              # mm/s
p = r["p"]
holes = case["holes"]
Q = case["q_ml_min"]
A = case["lumen_area_mm2"]
U = Q * 1e3 / 60 / A                                          # plug-flow speed of the irrigation alone, mm/s

g = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                        np.full(len(tets), pv.CellType.TETRA), xyz)
g["v"] = v
g["speed"] = np.linalg.norm(v, axis=1)
g["p"] = p
g.set_active_vectors("v")

# ------------------------------------------------------------------ flows through every boundary
X = xyz
An = 0.5 * np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])      # mm2, outward
q = np.einsum("ij,ij->i", v[tris].mean(axis=1), An) * 0.06                              # mL/min, + out
flows = {"irrigation_in": -q[lab == 1].sum(), "suction_out": q[lab == 2].sum()}
if holes:
    flows["bath_boundary_out"] = q[lab == 3].sum()
    gx, gz = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.5, 0.5, 21))
    inside = gx ** 2 + gz ** 2 < 0.25
    for k, (hx, hy, hz) in enumerate(holes):                  # flow up through each hole, probed inside the wall
        pts = np.c_[hx + gx[inside], hy + gz[inside], np.full(inside.sum(), hz - 0.3)]
        pr = pv.PolyData(pts).sample(g)
        flows[f"hole{k}_out"] = float(pr["v"][:, 2].mean() * np.pi * 0.25 * 0.06)
vol = np.abs(np.einsum("ij,ij->i", np.cross(X[tets[:, 1]] - X[tets[:, 0]], X[tets[:, 2]] - X[tets[:, 0]]),
                       X[tets[:, 3]] - X[tets[:, 0]])) / 6
cc = X[tets].mean(axis=1)
sp_c = g["speed"][tets].mean(axis=1)
in_lumen = (cc[:, 1] > YA) & (cc[:, 1] < YB) & (cc[:, 2] < -0.6)
V_lumen = vol[in_lumen].sum()

# ------------------------------------------------------------------ 1: two lengthwise planes
fig, axs = plt.subplots(2, 1, figsize=(18, 7.5))
vmax = max(np.percentile(g["speed"], 99.9), 1.0)
for ax, nrm, (i, j), lbl in ((axs[0], (1, 0, 0), (1, 2), "Z (mm)   [side view, plane through the axis]"),
                             (axs[1], (0, 0, 1), (1, 0), "X (mm)   [top view, plane through the axis]")):
    s = g.slice(normal=nrm, origin=(XC, 0, ZC)).triangulate()
    T = Triangulation(s.points[:, i], s.points[:, j], s.faces.reshape(-1, 4)[:, 1:])
    c = ax.tripcolor(T, np.maximum(s["speed"], vmax * 1e-4), norm=LogNorm(vmax * 1e-4, vmax), cmap="turbo", shading="gouraud")
    # in-plane arrows on a regular grid
    yy, ww = np.meshgrid(np.linspace(YA - 19, YB + 1, 76), np.linspace(s.points[:, j].min(), s.points[:, j].max(), 13))
    pts = np.zeros((yy.size, 3))
    pts[:, 0], pts[:, 2] = XC, ZC
    pts[:, 1] = yy.ravel()
    pts[:, j] = ww.ravel()
    pr = pv.PolyData(pts).sample(g)
    ok = pr["vtkValidPointMask"] > 0
    vv = pr["v"][ok]
    sp = np.linalg.norm(vv, axis=1) + 1e-12
    ax.quiver(pts[ok, i], pts[ok, j], vv[:, i] / sp, vv[:, j] / sp, color="w", scale=90, width=0.0012, alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("Y along the drain (mm): suction end on the left, tip with the irrigation outlet on the right")
    ax.set_ylabel(lbl)
    fig.colorbar(c, ax=ax, label="speed (mm/s)", pad=0.01)
fig.suptitle(f"{name}: speed (log scale) and flow direction. Irrigation {Q:g} mL/min; "
             f"out through suction {flows['suction_out']:.1f} mL/min", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(out, "planes.png"), dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ 2: cross-sections, axial velocity
ys = [55, 45, 30, 0, -30, -55]
fig, axs = plt.subplots(2, 3, figsize=(18, 8.5))
for ax, y in zip(axs.ravel(), ys):
    s = g.slice(normal=(0, 1, 0), origin=(0, y, 0)).triangulate()
    T = Triangulation(s.points[:, 0], s.points[:, 2], s.faces.reshape(-1, 4)[:, 1:])
    w = -s["v"][:, 1]                                          # towards suction = positive
    lim = max(np.percentile(np.abs(w), 99.5), 1e-6)
    c = ax.tripcolor(T, w, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="gouraud")
    ax.plot(np.r_[prof[:, 0], prof[0, 0]], np.r_[prof[:, 1], prof[0, 1]], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"Y = {y} mm  ({YB - y:.0f} mm from the tip)")
    fig.colorbar(c, ax=ax, label="speed towards suction (mm/s)", shrink=0.8)
fig.suptitle(f"{name}: flow along the drain in six cross-sections (red = towards suction, blue = back towards the tip)")
fig.tight_layout()
fig.savefig(os.path.join(out, "sections.png"), dpi=100)
plt.close(fig)

# ------------------------------------------------------------------ 3: streamlines from the irrigation outlet
rng = np.random.default_rng(0)
n_seed = 400
rr, th = 0.45 * np.sqrt(rng.uniform(0, 1, n_seed)), rng.uniform(0, 2 * np.pi, n_seed)
seed = pv.PolyData(np.c_[XC + rr * np.cos(th), np.full(n_seed, YB + 1.0), ZC + rr * np.sin(th)])
sl = g.streamlines_from_source(seed, vectors="v", integration_direction="forward", max_length=4000.0,
                               initial_step_length=0.1, max_step_length=0.5, max_steps=40000, terminal_speed=1e-5)
ends = {"suction": 0, "stalled": 0}
ends.update({f"hole{k}": 0 for k in range(len(holes))})
t_out = []
for i in range(sl.n_cells):
    ids = sl.get_cell(i).point_ids
    e = sl.points[ids[-1]]
    if e[1] < YA - 17:
        ends["suction"] += 1
        t_out.append(sl["IntegrationTime"][ids[-1]])
    else:
        hit = [int(np.argmin([np.hypot(e[0] - hx, e[1] - hy) for hx, hy, hz in holes]))] if holes and e[2] > -0.5 else []
        ends[f"hole{hit[0]}" if hit else "stalled"] += 1
pl = pv.Plotter(off_screen=True, window_size=(2200, 900))
wall = g.extract_surface(algorithm="dataset_surface")
pl.add_mesh(wall.clip(normal=(0, 0, 1), origin=(0, 0, ZC + 0.1), invert=True), color="#c9d2dd", opacity=0.35)
if sl.n_points:
    pl.add_mesh(sl.tube(radius=0.05), scalars="speed", cmap="turbo", log_scale=True, clim=[max(vmax * 1e-3, 1e-3), vmax],
                scalar_bar_args=dict(title="speed (mm/s)"))
for k, (hx, hy, hz) in enumerate(holes):
    pl.add_mesh(pv.Sphere(radius=0.6, center=(hx, hy, hz + 0.6)), color="#e02f2f")
pl.add_text(f"{name}: paths of the irrigation fluid from the tip outlet (right) to the suction tube (left)", font_size=12)
pl.camera_position = [(XC + 25, 0, ZC + 150), (XC, 0, ZC), (1, 0, 0)]
pl.camera.zoom(2.4)
pl.screenshot(os.path.join(out, "streamlines.png"))
pl.close()

# ------------------------------------------------------------------ 4: flushing: where does the fluid at each place come from, and how old is it
n_p = 3000
pick = rng.choice(np.nonzero(in_lumen)[0], n_p, p=vol[in_lumen] / V_lumen)
src = pv.PolyData(cc[pick])
T_MAX = 3600.0
bk = g.streamlines_from_source(src, vectors="v", integration_direction="backward", max_length=1e5,
                               initial_step_length=0.1, max_step_length=0.5, max_steps=60000, terminal_speed=1e-7)
age = np.full(n_p, np.inf)
origin = np.full(n_p, -1)                                     # 0 irrigation, 1+k hole k, -1 not reached
sid = bk["SeedIds"] if "SeedIds" in bk.cell_data else np.arange(bk.n_cells)
for i in range(bk.n_cells):
    ids = bk.get_cell(i).point_ids
    e = bk.points[ids[-1]]
    t = abs(bk["IntegrationTime"][ids[-1]])
    if e[1] > YB - 0.3 and np.hypot(e[0] - XC, e[2] - ZC) < 0.7:
        age[sid[i]], origin[sid[i]] = t, 0
    elif holes and e[2] > -0.5:
        age[sid[i]], origin[sid[i]] = t, 1
frac_irr = float((origin == 0).mean())
frac_bath = float((origin > 0).mean())
frac_none = float((origin < 0).mean())
tn = V_lumen / (flows["suction_out"] * 1e3 / 60)              # nominal residence time, s
fig, axs = plt.subplots(1, 2, figsize=(18, 5.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
P = cc[pick]
col = np.where(origin == 0, "#1f77b4", np.where(origin > 0, "#d62728", "#444444"))
ax.scatter(P[:, 1], P[:, 0], s=5, c=col, alpha=0.7)
for k, (hx, hy, hz) in enumerate(holes):
    ax.plot(hy, hx, "o", ms=9, mfc="none", mec="k")
ax.set_aspect("equal")
ax.set_xlabel("Y along the drain (mm), tip on the right")
ax.set_ylabel("X (mm)")
ax.set_title("what fills each place in the lumen (top view)\nblue = irrigation fluid, red = fluid drawn in through the open holes, "
             "grey = neither arrives (stagnant)")
ax = axs[1]
ts = np.sort(age[np.isfinite(age)])
ax.plot(ts, np.arange(1, len(ts) + 1) / n_p * 100, "-", lw=2)
ax.axvline(tn, color="k", ls="--", lw=1)
ax.text(tn, 5, " volume / flow", rotation=90, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("time since the fluid entered (s)")
ax.set_ylabel("% of the lumen volume reached by then")
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
ax.set_title("how fast the lumen is renewed (advection only)")
fig.tight_layout()
fig.savefig(os.path.join(out, "flushing.png"), dpi=110)
plt.close(fig)

pc = lambda t: float((age <= t).mean())
summary = dict(case=name, irrigation_ml_min=Q, viscosity_mPas=case["mu"] * 1e3, nodes=case["nodes"], tets=case["tets"],
               solver_step=int(r["step"]), flows_ml_min={k: round(float(x), 3) for k, x in flows.items()},
               lumen_volume_ml=round(float(V_lumen) / 1e3, 2), nominal_residence_s=round(float(tn), 1),
               plug_speed_mm_s=round(float(U), 3),
               pressure_tip_Pa=round(float(p[(X[:, 1] > YB - 3) & (X[:, 1] < YB)].mean()), 1),
               pressure_mid_Pa=round(float(p[np.abs(X[:, 1]) < 2].mean()), 1),
               slow_volume_fraction=round(float(vol[in_lumen & (sp_c < 0.1 * U)].sum() / V_lumen), 3),
               irrigant_streamline_ends={k: round(n / max(sl.n_cells, 1), 3) for k, n in ends.items()},
               irrigant_transit_s=dict(fastest=round(float(np.min(t_out)), 1) if t_out else None,
                                       median=round(float(np.median(t_out)), 1) if t_out else None),
               lumen_filled_by=dict(irrigation=round(frac_irr, 3), holes=round(frac_bath, 3), not_reached=round(frac_none, 3)),
               lumen_renewed_within={"1x_residence": round(pc(tn), 3), "3x_residence": round(pc(3 * tn), 3),
                                     "10x_residence": round(pc(10 * tn), 3)})
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

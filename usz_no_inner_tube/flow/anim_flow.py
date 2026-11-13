"""3D animation of a solved case: dye released into the frozen flow field, camera swinging round the sheath.

  python anim_flow.py runs/<case> [--tmax 120] [--power 2.5] [--frames 80]

Writes results/<case>/flow3d.gif.  Irrigation dye starts at the tip outlet; in the open-hole cases a second dye
starts in the bath just above every open hole.  The flow field is the last saved solver step (frozen), the clock
on screen is the time since the dye was released.
"""
import argparse, json, os
import numpy as np
import pyvista as pv
import imageio.v2 as imageio

HERE = os.path.dirname(os.path.abspath(__file__))
XC, ZC = 0.78, -4.82
YB = 59.9
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--tmax", type=float, default=120.0)
ap.add_argument("--power", type=float, default=2.5)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--title", default="")
ap.add_argument("--particles", action="store_true", help="moving tracer particles instead of a dye front")
ap.add_argument("--dt", type=float, default=0.00025)
a = ap.parse_args()
name = os.path.basename(os.path.normpath(a.run))
out = os.path.join(HERE, "results", name)
os.makedirs(out, exist_ok=True)
case = json.load(open(os.path.join(a.run, "case.json")))
m = np.load(os.path.join(a.run, "mesh.npz"))
r = np.load(os.path.join(a.run, "result.npz"))
xyz, tets, tris, lab = m["xyz"], m["tets"], m["tris"], m["lab"]
holes = case["holes"]

g = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                        np.full(len(tets), pv.CellType.TETRA), xyz)
g["v"] = r["v"] * 1e3                                          # mm/s
g.set_active_vectors("v")

# wall surface without the flat floor of the bath
wt = tris[lab == 0]
z = xyz[wt][:, :, 2]
wt = wt[~((z.min(1) > 0.05) & (np.ptp(z, axis=1) < 1e-3))]
wall = pv.PolyData(xyz, np.hstack([np.full((len(wt), 1), 3), wt]).ravel()).clean()


def trace(pts):
    s = g.streamlines_from_source(pv.PolyData(pts), vectors="v", integration_direction="forward", max_length=4000.0,
                                  initial_step_length=0.1, max_step_length=0.4, max_steps=40000, terminal_speed=1e-5)
    return np.asarray(s.points), np.asarray(s["IntegrationTime"]), np.linalg.norm(np.asarray(s["v"]), axis=1)


rng = np.random.default_rng(1)

def track(nf, tmax, dt):
    """Tracers released continuously at the irrigation outlet and in every open hole; explicit steps in the frozen field."""
    X = xyz[tets].astype(float)
    Minv = np.linalg.inv(np.transpose(X[:, 1:] - X[:, :1], (0, 2, 1)))
    vn = np.asarray(g["v"], float)
    nsub = max(int(round(tmax / nf / dt)), 1)
    P, K = np.zeros((0, 3)), np.zeros(0, int)
    out_f = []
    for f in range(nf):
        for s_ in range(nsub):
            if s_ % 20 == 0:
                rr, th = 0.45 * np.sqrt(rng.uniform(0, 1, 5)), rng.uniform(0, 2 * np.pi, 5)
                new, kind = [np.c_[XC + rr * np.cos(th), np.full(5, YB + 1.0), ZC + rr * np.sin(th)]], [np.zeros(5, int)]
                for hx, hy, hz in holes:
                    rr, th = 0.42 * np.sqrt(rng.uniform(0, 1, 5)), rng.uniform(0, 2 * np.pi, 5)
                    new.append(np.c_[hx + rr * np.cos(th), hy + rr * np.sin(th), np.full(5, 0.15)])
                    kind.append(np.ones(5, int))
                P, K = np.vstack([P] + new), np.concatenate([K] + kind)
            c = np.asarray(g.find_containing_cell(P))
            ok = c >= 0                                               # seeds are inside; everything kept is inside
            P, K, c = P[ok], K[ok], c[ok]
            lam = np.einsum("nij,nj->ni", Minv[c], P - X[c, 0])
            w = np.c_[1 - lam.sum(1), lam]
            Pn = P + np.einsum("ni,nij->nj", w, vn[tets[c]]) * dt + rng.normal(0, 0.02, P.shape)   # small random walk: frees tracers from the wall cell layer
            gone = Pn[:, 1] < -78.8                                   # left through the suction tube
            bad = (np.asarray(g.find_containing_cell(Pn)) < 0) & ~gone
            if bad.any():                                             # overshot a wall: step back with a small random nudge
                trial = P[bad] + rng.normal(0, 0.12, (int(bad.sum()), 3))
                good = np.asarray(g.find_containing_cell(trial)) >= 0
                Pn[bad] = np.where(good[:, None], trial, P[bad])
            P, K = Pn[~gone], K[~gone]
        out_f.append((P.copy(), K.copy()))
        if f % 10 == 0:
            print("tracked frame", f, "particles", len(P), flush=True)
    return out_f

n = 260
rr, th = 0.45 * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * np.pi, n)
TR = track(a.frames, a.tmax, a.dt) if a.particles else None
P_irr, T_irr, S_irr = trace(np.c_[XC + rr * np.cos(th), np.full(n, YB + 1.0), ZC + rr * np.sin(th)])
P_bath = T_bath = None
if holes and not a.particles:
    seeds = []
    for hx, hy, hz in holes:
        rr, th = 1.3 * np.sqrt(rng.uniform(0, 1, 90)), rng.uniform(0, 2 * np.pi, 90)
        seeds.append(np.c_[hx + rr * np.cos(th), hy + rr * np.sin(th), np.full(90, hz + 1.6)])
    P_bath, T_bath, _ = trace(np.vstack(seeds))
print("dye points", len(P_irr), 0 if P_bath is None else len(P_bath))

W, H = 1200, 430
pl = pv.Plotter(off_screen=True, window_size=(W, H))
pl.set_background("#0d1220")
pl.add_mesh(wall, color="#aebccd", opacity=0.13, smooth_shading=True)
for hx, hy, hz in holes:
    pl.add_mesh(pv.Disc(center=(hx, hy, hz + 0.35), inner=0.55, outer=0.85, normal=(0, 0, 1)), color="#ff5a4d")
pl.add_mesh(pv.Cone(center=(XC, YB + 3.2, ZC), direction=(0, -1, 0), height=2.4, radius=0.9), color="#35d6ff")
title = a.title or name
pl.add_text(title, position=(14, H - 34), font_size=11, color="white")
pl.add_text("irrigation outlet (tip)", position=(14, H - 150), font_size=8, color="#35d6ff")
pl.add_text("to suction", position=(W - 110, H - 150), font_size=8, color="#c8d3e3")
legend = "blue: irrigation fluid" + ("     orange: fluid drawn in through the open holes" if holes
                                     else "  (colour = speed, red fast, blue slow)")
pl.add_text(legend, position=(14, 10), font_size=9, color="#c8d3e3")
pl.camera.parallel_projection = True
focal = np.array([XC, -9.0, ZC])
clim = [0.3, max(float(S_irr.max()), 1.0)]

frames = []
nf, hold = a.frames, (0 if a.particles else 12)
actors = []
for f in range(nf + hold):
    u = min(f / (nf - 1), 1.0)
    t = a.tmax * (f + 1) / nf if a.particles else a.tmax * u ** a.power
    phi = np.radians(38.0 * np.sin(2 * np.pi * f / (nf + hold)) + 18.0)
    pl.camera.position = focal + 300.0 * np.array([np.sin(phi), 0.12, np.cos(phi)])
    pl.camera.focal_point = focal
    pl.camera.up = (np.cos(phi), 0, -np.sin(phi))
    pl.camera.parallel_scale = 27.0
    for ac in actors:
        pl.remove_actor(ac)
    actors = []
    if a.particles:
        Pf, Kf = TR[f]
        for kk, col, sz in ((1, '#ffa32b', 3.0), (0, '#35d6ff', 4.5)):
            if (Kf == kk).any():
                actors.append(pl.add_mesh(pv.PolyData(Pf[Kf == kk]), color=col, point_size=sz, render_points_as_spheres=True))
    k = (T_irr <= t) & (not a.particles)
    if k.any():
        d = pv.PolyData(P_irr[k])
        if holes:
            actors.append(pl.add_mesh(d, color="#35d6ff", point_size=3.5, render_points_as_spheres=True))
        else:
            d["speed (mm/s)"] = np.clip(S_irr[k], clim[0], None)
            actors.append(pl.add_mesh(d, scalars="speed (mm/s)", cmap="turbo", log_scale=True, clim=clim, point_size=3.5,
                                      render_points_as_spheres=True, show_scalar_bar=False))
    if P_bath is not None:
        k = T_bath <= t
        if k.any():
            actors.append(pl.add_mesh(pv.PolyData(P_bath[k]), color="#ffa32b", point_size=3.0,
                                      render_points_as_spheres=True, opacity=0.8))
    actors.append(pl.add_text(f"time since dye release  {t:6.2f} s" if a.tmax < 20 else
                              f"time since dye release  {t:5.1f} s", position=(W - 330, 10), font_size=11,
                              color="#ffd866", name="clock"))
    frames.append(pl.screenshot(return_img=True))
    if f % 10 == 0:
        print("frame", f, f"t={t:.2f}", flush=True)
pl.close()
path = os.path.join(out, "flow3d.gif")
imageio.mimsave(path, frames, duration=0.09, loop=0)
imageio.imwrite(os.path.join(out, "flow3d_last.png"), frames[nf - 1])
imageio.imwrite(os.path.join(out, "flow3d_mid.png"), frames[nf // 2])
print("written", path, round(os.path.getsize(path) / 1e6, 1), "MB")

"""3D pictures of a Kratos drain run: the drain cut open along its axis, in the world frame.

    python vof/kratos/render_3d.py <run dir> [--all] [--seconds 1.5]

Each picture shows, for one field output:
- the far half of the drain wall (and the trough floor) coloured by pressure,
- the liquid-air interfaces in 3D (pool surface, the level inside the drain, bubbles), translucent blue,
- streamlines of the flow in the drain and the tube, coloured by speed,
- the pressure on the cut plane through the axis.
Writes <run>/pictures/cut3d_<t>ms.png for the last output (or every output with --all) and,
with --all, cut3d_slow.gif.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_kratos as rk  # noqa: E402


def world(mesh, tilt):
    m = mesh.copy()
    m.points = m.points * 1e3
    m.rotate_y(-tilt, inplace=True)
    return m


def picture(g, st, t, path, title):
    import pyvista as pv
    pv.OFF_SCREEN = True
    tilt = rk.TILT
    g = g.copy()
    U = np.asarray(g.point_data["VELOCITY"])
    g.point_data["speed"] = np.linalg.norm(U, axis=1)
    # far half (y > 0) of the boundary: drain wall and trough floor/walls, coloured by pressure
    surf = g.extract_surface(algorithm="dataset_surface")
    cc = surf.cell_centers().points * 1e3
    far = cc[:, 1] > 0.0
    near_drain = (np.abs(cc[:, 1]) < 9.9) & (cc[:, 2] > -1.0) & (cc[:, 2] < 11.2) & (cc[:, 0] > -1.0)
    walls = surf.extract_cells(np.where(far & near_drain)[0])
    floor = surf.extract_cells(np.where(far & ~near_drain & (cc[:, 2] < -2.0))[0])
    # cut plane through the axis (pressure)
    cut = g.slice(normal=(0, 1, 0), origin=(0, 0, 0))
    # liquid-air interfaces (only the near half is clipped away so the inside is visible)
    iface = g.contour([0.0], scalars="DISTANCE").clip(normal=(0, -1, 0), origin=(0, 0, 0), invert=False)
    # streamlines seeded along the inside of the drain and near the submerged openings
    seeds = pv.PolyData(np.column_stack([np.linspace(0.004, 0.117, 60), np.zeros(60),
                                         np.full(60, 0.0048)]))
    seeds2 = pv.PolyData(np.column_stack([np.linspace(0.004, 0.117, 40), np.full(40, -0.003),
                                          np.full(40, 0.002)]))
    try:
        sl = g.streamlines_from_source(seeds.merge(seeds2), vectors="VELOCITY", max_steps=4000,
                                       integration_direction="both", initial_step_length=0.2,
                                       max_step_length=0.5)
    except Exception:                               # noqa: BLE001
        sl = None
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1150))
    pl.set_background("white")
    pbar = dict(vertical=True, height=0.32, width=0.03, title_font_size=17, label_font_size=14, color="black",
                fmt="%.0f")
    pl.add_mesh(world(walls, tilt), scalars="PRESSURE", cmap="RdBu_r", clim=(-950, 950), opacity=0.55,
                scalar_bar_args=dict(title="pressure (Pa)", position_x=0.91, position_y=0.6, **pbar))
    pl.add_mesh(world(floor, tilt), color="#d9d7cf", opacity=0.35)
    pl.add_mesh(world(cut, tilt), scalars="PRESSURE", cmap="RdBu_r", clim=(-950, 950), opacity=0.9,
                show_scalar_bar=False)
    if iface.n_points:
        pl.add_mesh(world(iface, tilt), color="#1f6fd1", opacity=0.45, smooth_shading=True)
    if sl is not None and sl.n_points:
        pl.add_mesh(world(sl, tilt), scalars="speed", cmap="magma", clim=(0, 3.0), line_width=2.5,
                    scalar_bar_args=dict(title="speed (m/s)", position_x=0.91, position_y=0.15,
                                         vertical=True, height=0.32, width=0.03, title_font_size=17,
                                         label_font_size=14, color="black", fmt="%.1f"))
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.camera_position = [(45, -205, 95), (66, 0, 13), (0, 0, 1)]
    pl.camera.zoom(1.55)
    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(path)
    pl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--all", action="store_true", help="every output and an animated GIF")
    ap.add_argument("--seconds", type=float, default=1.5)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    st = json.load(open(os.path.join(run, "model_state.json")))
    mon = rk.read_monitor(run)
    files = rk.vtk_files(run)
    if not a.all:
        files = files[-1:]
    label = "USZ as built" if st["case"].startswith("usz") else "Recommended"
    out = os.path.join(run, "pictures")
    frames = []
    for f in files:
        t = rk.vtk_time(f)
        i = int(np.argmin(np.abs(mon["t"] - t)))
        ql = mon["Q_liquid"][i] * 6e7
        qg = (mon["Q_total"][i] - mon["Q_liquid"][i]) * 6e4
        title = (f"{label} drain, pool {st['h']:g} mm, outlet end {rk.TILT:g} deg up, t = {t * 1e3:.1f} ms  |  "
                 f"liquid out {ql:.0f} mL/min, air out {qg:.1f} L/min\n"
                 "cut open along the axis: far half of the wall and the cut plane coloured by pressure, "
                 "liquid surfaces in blue, streamlines coloured by speed")
        p = os.path.join(out, f"cut3d_{t * 1e3:05.1f}ms.png")
        picture(rk.load(f), st, t, p, title)
        frames.append(p)
        print("  wrote", os.path.basename(p))
    if a.all and len(frames) > 1:
        import imageio.v2 as imageio
        from PIL import Image
        imgs = [Image.open(p).convert("RGB") for p in frames]
        w = 1400
        imgs = [np.asarray(im.resize((w, int(im.height * w / im.width)))) for im in imgs]
        dur = [a.seconds * 1000] * (len(imgs) - 1) + [a.seconds * 3000]
        imageio.mimsave(os.path.join(out, "cut3d_slow.gif"), imgs, duration=dur, loop=0)
        print("  wrote cut3d_slow.gif")


if __name__ == "__main__":
    main()

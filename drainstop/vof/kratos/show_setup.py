"""Pictures of what a Kratos case simulates: geometry, boundary conditions, start and mesh.

    python vof/kratos/show_setup.py <run dir>

Writes <run>/pictures/setup_3d.png (drain, openings, tube, trough, starting liquid, boundary
conditions, world frame) and setup_mesh.png (cuts through the tetrahedral mesh).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_case import SPECS, TILT, initial_distance, run_study  # noqa: E402


def read_msh(path):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(path)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    idx = np.full(int(tags.max()) + 1, -1)
    idx[tags.astype(int)] = np.arange(tags.size)
    et, _, en = gmsh.model.mesh.getElements(3)
    tets = idx[np.concatenate([e for t, e in zip(et, en) if t == 4]).astype(int)].reshape(-1, 4)
    groups = {}
    for dim, ptag in gmsh.model.getPhysicalGroups(2):
        name = gmsh.model.getPhysicalName(dim, ptag)
        tris = []
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
            et, _, en = gmsh.model.mesh.getElements(2, ent)
            tris += [idx[e.astype(int)].reshape(-1, 3) for t, e in zip(et, en) if t == 2]
        groups[name] = np.concatenate(tris)
    gmsh.finalize()
    return xyz, tets, groups


def main():
    import pyvista as pv
    pv.OFF_SCREEN = True
    run = os.path.abspath(sys.argv[1])
    st = json.load(open(os.path.join(run, "model_state.json")))
    info = json.load(open(os.path.join(run, "mesh_info.json")))
    m = run_study.make_model(SPECS[st["case"]] or dict(run_study.best_spec(), tilt=TILT))
    xyz, tets, groups = read_msh(os.path.join(run, "drain.msh"))
    out = os.path.join(run, "pictures")
    os.makedirs(out, exist_ok=True)
    grid = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                               np.full(len(tets), pv.CellType.TETRA), xyz)
    grid["d"] = initial_distance(m, xyz, st["h"], st["zeta"]) * 1e3

    def surf(name):
        t = groups[name]
        return pv.PolyData(xyz, np.hstack([np.full((len(t), 1), 3), t]).ravel()).clean()
    walls = surf("Walls")
    cc = walls.cell_centers().points
    a, b = m.drain.a, m.drain.b
    on_drain = (np.abs(cc[:, 1]) < a + 1.0) & (cc[:, 2] > -1.0) & (cc[:, 2] < 2 * b + 1.5) & (cc[:, 0] > -1.0)
    drain = walls.extract_cells(np.where(on_drain)[0]).extract_surface()
    trough = walls.extract_cells(np.where(~on_drain)[0]).extract_surface()
    outlet, atm = surf("Outlet"), surf("Atmosphere")
    liquid = grid.contour([0.0], scalars="d")
    pool = grid.threshold(0.0, scalars="d", invert=True)

    def world(mesh):
        mesh = mesh.copy()
        mesh.rotate_y(-TILT, inplace=True)
        return mesh

    pl = pv.Plotter(off_screen=True, window_size=(2200, 1100))
    pl.set_background("white")
    pl.add_mesh(world(pool.extract_surface(algorithm="dataset_surface")), color="#1f6fd1", opacity=0.75,
                smooth_shading=True)
    pl.add_mesh(world(drain), color="#bdbbb3", opacity=0.16, smooth_shading=True)
    pl.add_mesh(world(drain).extract_feature_edges(25), color="#3d3c39", line_width=1.2)
    pl.add_mesh(world(grid.outline()), color="#5f5e5a", line_width=1.5)
    pl.add_mesh(world(outlet), color="#c43b3a")
    al = math.radians(TILT)
    L = m.drain.length

    def wpt(x, y, z):
        return (x * math.cos(al) - z * math.sin(al), y, x * math.sin(al) + z * math.cos(al))
    zt = info["z_top"]
    pl.add_point_labels([wpt(L + 16.8, 0, b), wpt(60, 0, zt), wpt(8, -14, -2.5),
                         wpt(95, 0, 2 * b - 1.5), wpt(30, -a, 0.6)],
                        ["outlet: suction -933 Pa (-7 mmHg)", "open top: air at 0 Pa",
                         f"pool (liquid) up to {st['h']:g} mm", f"liquid inside the drain up to {st['zeta']:.1f} mm",
                         "openings (1.0 mm)"],
                        font_size=20, text_color="black", shape_opacity=0.8, shape_color="white",
                        point_size=8, point_color="black", always_visible=True)
    pl.add_arrows(np.array([[-8.0, -20.0, 38.0]]), np.array([[0.0, 0.0, -12.0]]), color="black")
    pl.add_point_labels([(-8.0, -20.0, 40.0)], ["gravity"], font_size=18, text_color="black",
                        shape_opacity=0.0, point_size=1, always_visible=True)
    label = "USZ as built" if st["case"].startswith("usz") else "Recommended layout"
    pl.add_text(f"What is simulated: {label}, outlet end {TILT:g} deg up, {m.pattern.n} openings drilled "
                f"through the 0.6 mm wall; trough around the drain; blue = liquid at the start",
                position="upper_left", font_size=12, color="black")
    pl.camera_position = [(62, -260, 75), (71, 0, 14), (0, 0, 1)]   # from the side, a little above
    pl.camera.zoom(1.6)
    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(os.path.join(out, "setup_3d.png"))
    pl.close()

    # mesh cuts: plane through the axis near the outlet end, and a cross-section through openings
    pat = m.pattern
    near = (np.sin(pat.theta) <= 0)
    xh = float(pat.x[near][np.argmin(np.abs(pat.x[near] - 80))])
    cuts = [((0, 1, 0), (0, 0, 0), "plane through the drain axis, outlet end", (88, 137, -3, 12)),
            ((1, 0, 0), (xh, 0, 0), f"cross-section through openings at x = {xh:.1f} mm", None)]
    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(2400, 900), border=False)
    for k, (n, o, title, crop) in enumerate(cuts):
        pl.subplot(0, k)
        pl.set_background("white")
        sl = grid.slice(normal=n, origin=o)
        pl.add_mesh(sl, scalars="d", cmap=["#2a78d6", "#f4f3ee"], clim=(-0.01, 0.01), show_edges=True,
                    edge_color="#5f5e5a", line_width=0.4, show_scalar_bar=False)
        pl.add_text(title, position="upper_left", font_size=12, color="black")
        if k == 0:
            pl.view_xz()
            pl.camera.position = (112, -200, 5)
            pl.camera.focal_point = (112, 0, 5)
            pl.camera.zoom(4.2)
        else:
            pl.view_yz()
            pl.camera.zoom(1.6)
    pl.add_text(f"{info['nodes']:,} nodes, {info['tets']:,} tetrahedra; {info['hole']} mm cells at the opening "
                "rims and the bore (blue = liquid at the start)", position="lower_left", font_size=11,
                color="black")
    pl.screenshot(os.path.join(out, "setup_mesh.png"))
    pl.close()
    print("wrote", os.path.join(out, "setup_3d.png"), os.path.join(out, "setup_mesh.png"))


if __name__ == "__main__":
    main()

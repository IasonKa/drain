"""
Pictures of the geometry:
  01_drain_with_sponge_organ_<design>.png   whole drain + sponge organ lying on it
  02_hole_window_cutaway_<design>.png       the single-hole window cut open
  03_mesh_section_<design>.png              the structured mesh shared by OpenFOAM and FEBio
  04_section_at_hole_<design>.png           section through the real drain CAD at the hole

python scripts/render_cad.py [--design standoff]   (drain STLs in data/, CAD from scripts/build_cad.py)
"""
import argparse, json, os, sys
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge.geom import Window, WindowMesh

PINK, DARKPINK, BLUE, GREY = "#e8a0a8", "#9c3848", "#4a90d9", "#8a9bb0"


def sponge_scalars(mesh, freq=0.9, seed=1):
    """Porous look: thresholded Perlin noise -> dark 'pores' on pink tissue."""
    f = pv.perlin_noise(1.0, (freq, freq, freq), (seed, 2 * seed, 3 * seed))
    v = np.array([f.EvaluateFunction(p) for p in mesh.points])
    f2 = pv.perlin_noise(1.0, (3 * freq, 3 * freq, 3 * freq), (5, 7, 11))
    v += 0.5 * np.array([f2.EvaluateFunction(p) for p in mesh.points])
    return v


def cut_dirs(w):
    """Section plane through the hole axis: along X for the current design; at 45 deg for ring
    designs, so that the cut goes through two ring arcs instead of along a slot."""
    if w.has_standoff:
        c = np.sqrt(0.5)
        return np.array([c, c, 0.0]), np.array([-c, c, 0.0])
    return np.array([1.0, 0, 0]), np.array([0, 1.0, 0])


def _loops(mesh, T, e=np.array([1.0, 0, 0]), n=np.array([0, 1.0, 0])):
    """Slice a closed surface with the plane through the hole axis with normal n (local frame)
    -> list of closed 2D loops (coordinate along e, local z)."""
    O = T[:3, 3]
    cut = mesh.slice(normal=tuple(T[:3, :3] @ n), origin=tuple(O))
    Ti = np.linalg.inv(T)
    loops = []
    for body in cut.connectivity().split_bodies():          # trace each closed outline on its own
        sl = body.extract_geometry().strip(join=True)
        lines = sl.lines
        i = 0
        while i < len(lines):
            k = lines[i]
            ids = lines[i + 1:i + 1 + k]
            i += k + 1
            P = sl.points[ids] @ Ti[:3, :3].T + Ti[:3, 3]
            if k > 3:
                loops.append(np.c_[P @ e, P[:, 2]])
    return loops


def _compound(loops):
    """Matplotlib path with holes: orient loops by nesting depth (non-zero rule)."""
    from matplotlib.path import Path
    verts, codes = [], []
    paths = [Path(l) for l in loops]
    for i, l in enumerate(loops):
        depth = sum(paths[j].contains_point(l[0]) for j in range(len(loops)) if j != i)
        area = 0.5 * np.sum(l[:-1, 0] * l[1:, 1] - l[1:, 0] * l[:-1, 1])
        if (depth % 2 == 0) != (area > 0):
            l = l[::-1]
        verts += list(l) + [l[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(l) - 1) + [Path.CLOSEPOLY]
    return Path(np.array(verts), codes)


def section_figure(drain, organ, T, w, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Rectangle
    e, n = cut_dirs(w)
    dl, ol = _loops(drain, T, e, n), _loops(organ, T, e, n)
    dname = "drain, standoff-ring design" if w.has_standoff else "drain, current dome-cap design"
    along = "distance from the hole axis along the 45 deg cut (through two ring arcs), mm" if w.has_standoff         else "lateral X (mm, hole frame)"
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, lim in ((axs[0], (-15, 15, -12, 9)), (axs[1], (-2.0, 2.0, -1.0, 0.55 + w.top_feature))):
        ax.add_patch(PathPatch(_compound(ol), fc=PINK, ec=DARKPINK, lw=0.8, label="sponge organ (biphasic)"))
        ax.add_patch(PathPatch(_compound(dl), fc=GREY, ec="#3d4a5c", lw=0.8, label=dname))
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
        ax.set_xlabel(along); ax.set_ylabel("hole axis Z (mm)")
    axs[0].add_patch(Rectangle((-2, -1), 4, 1.55, fill=False, ec="k", lw=1, ls="--"))
    axs[0].set_title("cross-section through the studied hole (drain axis into the page)")
    axs[0].legend(loc="lower left", fontsize=9)
    ax = axs[1]
    ax.axvspan(-w.a, w.a, ymin=0, ymax=1, color="#4a90d9", alpha=0.10)
    ax.annotate("liquid gap %.0f um" % ((w.gap + w.top_feature) * 1e3), xy=(1.2, w.skin_e(1.2) + (w.gap + w.top_feature) / 2),
                xytext=(0.6, 0.38 + w.top_feature),
                arrowprops=dict(arrowstyle="->"), fontsize=10)
    ax.annotate("hole d = %.1f mm (suction)" % w.hole_d, xy=(0, -0.3), ha="center", fontsize=10)
    ax.set_title("zoom: the single-hole window (dashed box)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def window_blocks(w):
    m = WindowMesh(w)
    cells = np.hstack([np.full((len(m.cells), 1), 8), m.cells]).ravel()
    g = pv.UnstructuredGrid(cells, np.full(len(m.cells), pv.CellType.HEXAHEDRON), m.points)
    g.cell_data["zone"] = m.cell_zone
    return m, g


def render_design(design, a):
    from drainsponge import config
    from drainsponge.geom import Z_CHAN, Z_HOLE, Z_STAND, Z_FILM, Z_TIS
    w = Window.from_dict(config.load(design=design)["window"])
    cad = os.path.join(a.cad, design)
    sfx = f"_{design}"
    T = np.array(json.load(open(os.path.join(cad, "frame.json")))["local_to_global"])
    O, X, Y, Z = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]
    # full-drain CAD exists only for the two real designs (no full drain with the optimized rings yet)
    stl = os.path.join(a.data, {"current": "drainage-3D_cap_dome.stl",
                                "standoff": "drainage-3D_cap_dome_standoff.stl"}.get(design, "none.stl"))
    # ---------------------------------------------------------- 1 + 4: drain + organ, section
    if os.path.exists(stl):
        drain = pv.read(stl)
        organ = pv.read(os.path.join(cad, "organ_sponge_global.stl")).compute_normals()
        organ["s"] = sponge_scalars(organ, 0.35)
        p = pv.Plotter(off_screen=True, window_size=(1300, 1100))
        p.add_mesh(drain, color=GREY, smooth_shading=True, specular=0.3)
        p.add_mesh(organ, scalars="s", cmap=["#6e1f2c", "#b5525f", "#e59aa4", "#f4c6cb"], show_scalar_bar=False,
                   smooth_shading=True)
        lab = "standoff-ring drain" if w.has_standoff else "current dome-cap drain"
        p.add_text(f"{lab} + sponge organ (50 um above the {'ring tops' if w.has_standoff else 'skin'})", font_size=12)
        p.camera_position = [tuple(O + 120 * Z + 60 * X - 40 * Y), tuple(O - 10 * Y), tuple(Y)]
        p.camera.zoom(1.6)
        p.screenshot(os.path.join(a.out, f"01_drain_with_sponge_organ{sfx}.png"))
        p.close()
        section_figure(drain, organ, T, w, os.path.join(a.out, f"04_section_at_hole{sfx}.png"))
    else:
        print("drain STL not found, skipping pictures 1 and 4:", stl)
    # ---------------------------------------------------------- 2: window cut-away
    m, g = window_blocks(w)
    e, nn = cut_dirs(w)
    half = g.clip(normal=tuple(nn), origin=(0, 0, 0), invert=False, crinkle=False)
    colors = {Z_CHAN: "#bfe0ff", Z_HOLE: BLUE, Z_STAND: "#5aa0e6", Z_FILM: "#2f6db3", Z_TIS: PINK}
    p = pv.Plotter(off_screen=True, window_size=(1700, 1100))
    for zi, c in colors.items():
        part = half.extract_cells(half.cell_data["zone"] == zi)
        if part.n_cells == 0:
            continue
        surf = part.extract_surface(algorithm="dataset_surface")
        if zi == Z_TIS:
            surf["s"] = sponge_scalars(surf, 2.5)
            p.add_mesh(surf, scalars="s", cmap=["#b5525f", "#d9838e", "#efb0b8", "#f8d5d9"],
                       show_scalar_bar=False, opacity=0.95)
        else:
            p.add_mesh(surf, color=c, opacity=0.35 if zi == Z_CHAN else 0.9)
    wall = pv.read(os.path.join(cad, "window_drain.stl")).clip(normal=tuple(nn), origin=(0, 0, 0), invert=False)
    p.add_mesh(wall, color=GREY, smooth_shading=False, specular=0.2)
    hs = w.top_feature
    L1 = 1.6 if not hs else 1.95
    pts = [[*(L1 * e[:2]), hs + w.gap + 1.2], [*(L1 * e[:2]), hs + 0.03], [*(L1 * e[:2]), -0.3],
           [*(L1 * e[:2]), -1.1], [0, 0, -0.3]]
    labs = ["sponge tissue (biphasic)", "liquid gap" + (" 0.45 mm" if hs else " 50 um"), "drain wall 0.6 mm",
            "sheath channel -> suction", "hole d = 1 mm"]
    if hs:
        rm = 0.5 * (w.standoff_rin + w.standoff_rout)
        pts.append([*(-rm * e[:2]), hs + 0.02]); labs.append(f"standoff ring {hs:.2f} mm")
    p.add_point_labels(np.array(pts), labs, font_size=22, point_size=1, shape_opacity=0.7, always_visible=True)
    cam = 3.0 * e - 8.5 * nn
    p.camera_position = [(cam[0], cam[1], 1.6 + hs), (0, 0, -0.1 + 0.5 * hs), (0, 0, 1)]
    p.screenshot(os.path.join(a.out, f"02_hole_window_cutaway{sfx}.png"))
    p.close()
    # ---------------------------------------------------------- 3: mesh section + plan grid
    sl = g.clip(normal=tuple(nn), origin=(0, 0, 0), invert=False)
    p = pv.Plotter(off_screen=True, window_size=(1700, 900), shape=(1, 2))
    p.subplot(0, 0)
    p.add_mesh(sl, scalars="zone", cmap=["#bfe0ff", BLUE, "#5aa0e6", "#2f6db3", PINK], clim=[0, 4],
               show_edges=True, edge_color="#333333", line_width=0.3, show_scalar_bar=False)
    p.add_text("section through the hole (channel / hole / gap / tissue)", font_size=11)
    cam = -9 * nn
    p.camera_position = [(cam[0], cam[1], -0.1 + 0.4 * hs), (0, 0, -0.1 + 0.4 * hs), (0, 0, 1)]
    p.camera.zoom(1.3)
    p.subplot(0, 1)
    top = g.extract_cells(g.cell_data["zone"] == Z_FILM).extract_surface(algorithm="dataset_surface")
    p.add_mesh(top, color=BLUE, show_edges=True, edge_color="#112233", line_width=0.4)
    if hs:
        ring = g.extract_cells(np.isin(np.arange(g.n_cells), np.nonzero(
            (g.cell_data["zone"] == Z_FILM) & m.cell_ring[m.cell_plan])[0])).extract_surface(algorithm="dataset_surface")
        p.add_mesh(ring.translate((0, 0, 0.01)), color="#4a5568", show_edges=True, edge_color="#112233", line_width=0.4)
    p.add_text("plan O-grid (dark = ring arcs; slots and gap in blue)" if hs else "plan O-grid around the hole",
               font_size=11)
    p.camera_position = [(0, 0, 8), (0, 0, 0), (0, 1, 0)]
    p.screenshot(os.path.join(a.out, f"03_mesh_section{sfx}.png"))
    p.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=None, choices=["current", "standoff", "standoff_opt"])
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--cad", default="out/cad")
    ap.add_argument("--out", default="out/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for d in ([a.design] if a.design else ["current", "standoff", "standoff_opt"]):
        render_design(d, a)
        print("rendered", d)
    print("figures in", a.out)


if __name__ == "__main__":
    main()

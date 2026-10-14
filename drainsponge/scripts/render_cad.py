"""
Pictures of the geometry:
  01_drain_with_sponge_organ.png   whole dome-cap drain + sponge organ lying on it
  02_hole_window_cutaway.png       the single-hole window cut open (wall, hole, film, tissue, channel)
  03_mesh_section.png              the structured mesh shared by OpenFOAM and FEBio

python scripts/render_cad.py --drain <drainage-3D_cap_dome.stl> --out out/figures
"""
import argparse, json, os, sys
import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge.geom import Window, WindowMesh, ZONES

PINK, DARKPINK, BLUE, GREY = "#e8a0a8", "#9c3848", "#4a90d9", "#8a9bb0"


def sponge_scalars(mesh, freq=0.9, seed=1):
    """Porous look: thresholded Perlin noise -> dark 'pores' on pink tissue."""
    f = pv.perlin_noise(1.0, (freq, freq, freq), (seed, 2 * seed, 3 * seed))
    v = np.array([f.EvaluateFunction(p) for p in mesh.points])
    f2 = pv.perlin_noise(1.0, (3 * freq, 3 * freq, 3 * freq), (5, 7, 11))
    v += 0.5 * np.array([f2.EvaluateFunction(p) for p in mesh.points])
    return v


def _loops(mesh, T):
    """Slice a closed surface with the local XZ plane (y = 0) -> list of closed 2D loops (local x, z)."""
    O, Y = T[:3, 3], T[:3, 1]
    sl = mesh.slice(normal=tuple(Y), origin=tuple(O)).strip(join=True)
    Ti = np.linalg.inv(T)
    loops = []
    lines = sl.lines
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        i += n + 1
        P = sl.points[ids] @ Ti[:3, :3].T + Ti[:3, 3]
        if n > 3:
            loops.append(P[:, [0, 2]])
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
    dl, ol = _loops(drain, T), _loops(organ, T)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, lim in ((axs[0], (-15, 15, -12, 9)), (axs[1], (-2.0, 2.0, -1.0, 0.55))):
        ax.add_patch(PathPatch(_compound(ol), fc=PINK, ec=DARKPINK, lw=0.8, label="sponge organ (biphasic)"))
        ax.add_patch(PathPatch(_compound(dl), fc=GREY, ec="#3d4a5c", lw=0.8, label="drain (dome-cap design)"))
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
        ax.set_xlabel("lateral X (mm, hole frame)"); ax.set_ylabel("hole axis Z (mm)")
    axs[0].add_patch(Rectangle((-2, -1), 4, 1.55, fill=False, ec="k", lw=1, ls="--"))
    axs[0].set_title("cross-section through the studied hole (drain axis into the page)")
    axs[0].legend(loc="lower left", fontsize=9)
    ax = axs[1]
    ax.axvspan(-w.a, w.a, ymin=0, ymax=1, color="#4a90d9", alpha=0.10)
    ax.annotate("liquid film g0 = %.0f um" % (w.gap * 1e3), xy=(1.2, w.skin_e(1.2) + w.gap / 2), xytext=(0.6, 0.38),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drain", default=os.path.join(os.path.dirname(__file__), "..", "data", "drainage-3D_cap_dome.stl"))
    ap.add_argument("--cad", default="out/cad")
    ap.add_argument("--out", default="out/figures")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = json.load(open(a.config)) if a.config else {}
    w = Window.from_dict(cfg.get("window", {}))
    T = np.array(json.load(open(os.path.join(a.cad, "frame.json")))["local_to_global"])
    O, X, Y, Z = T[:3, 3], T[:3, 0], T[:3, 1], T[:3, 2]

    # ---------------------------------------------------------- 1: drain + organ
    if os.path.exists(a.drain):
        drain = pv.read(a.drain)
        organ = pv.read(os.path.join(a.cad, "organ_sponge_global.stl")).compute_normals()
        organ["s"] = sponge_scalars(organ, 0.35)
        p = pv.Plotter(off_screen=True, window_size=(1300, 1100))
        p.add_mesh(drain, color=GREY, smooth_shading=True, specular=0.3)
        p.add_mesh(organ, scalars="s", cmap=["#6e1f2c", "#b5525f", "#e59aa4", "#f4c6cb"], show_scalar_bar=False,
                   smooth_shading=True)
        p.add_text("dome-cap drain + sponge organ (50 um above the skin)", font_size=12)
        p.camera_position = [tuple(O + 120 * Z + 60 * X - 40 * Y), tuple(O - 10 * Y), tuple(Y)]
        p.camera.zoom(1.6)
        p.screenshot(os.path.join(a.out, "01_drain_with_sponge_organ.png"))
        p.close()
        section_figure(drain, organ, T, w, os.path.join(a.out, "04_section_at_hole.png"))
    else:
        print("drain STL not found, skipping picture 1:", a.drain)

    # ---------------------------------------------------------- 2: window cut-away
    m, g = window_blocks(w)
    half = g.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False, crinkle=False)
    colors = {0: "#bfe0ff", 1: BLUE, 2: "#2f6db3", 3: PINK}
    p = pv.Plotter(off_screen=True, window_size=(1700, 1100))
    for zi, c in colors.items():
        part = half.extract_cells(half.cell_data["zone"] == zi)
        if part.n_cells == 0:
            continue
        surf = part.extract_surface()
        if zi == 3:
            surf["s"] = sponge_scalars(surf, 2.5)
            p.add_mesh(surf, scalars="s", cmap=["#b5525f", "#d9838e", "#efb0b8", "#f8d5d9"],
                       show_scalar_bar=False, opacity=0.95)
        else:
            p.add_mesh(surf, color=c, opacity=1.0 if zi else 0.35)
    wall = pv.read(os.path.join(a.cad, "window_drain.stl")).clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
    p.add_mesh(wall, color=GREY, smooth_shading=False, specular=0.2)
    p.add_point_labels(np.array([[1.75, 0, w.gap + 1.2], [1.75, 0, 0.03], [1.75, 0, -0.3], [1.75, 0, -1.1], [0, 0, -0.3]]),
                       ["sponge tissue (biphasic)", "liquid film 50 um", "drain wall 0.6 mm",
                        "sheath channel -> suction", "hole d = 1 mm"],
                       font_size=22, point_size=1, shape_opacity=0.7, always_visible=True)
    p.camera_position = [(3.0, -8.5, 1.6), (0, 0, -0.1), (0, 0, 1)]
    p.screenshot(os.path.join(a.out, "02_hole_window_cutaway.png"))
    p.close()

    # ---------------------------------------------------------- 3: mesh section
    sl = g.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
    p = pv.Plotter(off_screen=True, window_size=(1700, 900), shape=(1, 2))
    cm = ["#bfe0ff", BLUE, "#2f6db3", PINK]
    p.subplot(0, 0)
    p.add_mesh(sl, scalars="zone", cmap=cm, show_edges=True, edge_color="#333333", line_width=0.3,
               show_scalar_bar=False)
    p.add_text("section through the hole (channel / hole / film / tissue)", font_size=11)
    p.camera_position = [(0, -9, -0.1), (0, 0, -0.1), (0, 0, 1)]
    p.camera.zoom(1.3)
    p.subplot(0, 1)
    top = g.extract_cells(g.cell_data["zone"] == 2).extract_surface()
    p.add_mesh(top, color=BLUE, show_edges=True, edge_color="#112233", line_width=0.4)
    p.add_text("plan O-grid around the hole (film layer)", font_size=11)
    p.camera_position = [(0, 0, 8), (0, 0, 0), (0, 1, 0)]
    p.screenshot(os.path.join(a.out, "03_mesh_section.png"))
    p.close()
    print("figures in", a.out)


if __name__ == "__main__":
    main()

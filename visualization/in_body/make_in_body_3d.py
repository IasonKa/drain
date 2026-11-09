"""
3D picture: the real standoff drain (your CAD) lying in the body among bowel loops, with an
irrigation catheter whose tip sits about 15 mm above the drain.

    py -3.12 make_in_body_3d.py            ->  in_body_3d.png  (+ in_body_3d_raw.png, no labels)

The drain is the real STL (drainsponge/data/drainage-3D_cap_dome_standoff.stl). Everything else
(organs, catheter, liquid) is an illustrative shape made here; sizes are realistic, the anatomy is
schematic. Units: mm. Frame: x along the drain (dome end at +x), y across it, z up (against gravity).
"""
import json, os, sys
import numpy as np
import pyvista as pv
import vtk

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "drainsponge", "data")
STL = os.path.join(DATA, "drainage-3D_cap_dome_standoff.stl")
CACHE = os.path.join(HERE, "_drain_body_cache.vtp")      # not committed (see .gitignore)

GREY = "#8a9bb0"
BLUE = "#2f7fd8"
SILICONE = "#f2eee0"
ORANGE = "#e8862a"
RNG = np.random.default_rng(3)


# ---------------------------------------------------------------- drain (real CAD)
def drain_body():
    """Real drain in the local frame, perforated body + dome + a short stub of the outlet tube."""
    if os.path.exists(CACHE):
        return pv.read(CACHE)
    m = pv.read(STL)
    hc = json.load(open(os.path.join(DATA, "hole_center.json")))
    Y = np.array(hc["global_Y_drain_axis"]); X = np.array(hc["global_X_lateral"]); Z = np.array(hc["global_Z_hole_axis"])
    P = m.points - np.array(hc["global_origin"])
    loc = np.c_[P @ Y, P @ X - 0.955, P @ Z + 4.815]          # centre the elliptical body on the axis
    m.points = loc
    r = np.hypot(loc[:, 1], loc[:, 2])
    keep = (loc[:, 0] > -81) & (np.abs(loc[:, 1]) < 10.5) & (np.abs(loc[:, 2]) < 6.0)
    keep &= (loc[:, 0] > -66) | (r < 3.3 + 0.45 * (loc[:, 0] + 81))   # taper only: drop the tube loop
    f = m.faces.reshape(-1, 4)[:, 1:]                          # STL: all triangles
    fk = keep[f].all(axis=1)                                   # (extract_points is far too slow here)
    body = pv.PolyData(m.points, np.c_[np.full(fk.sum(), 3), f[fk]].ravel()).clean()
    body = body.compute_normals(auto_orient_normals=False, split_vertices=False)
    body.save(CACHE)
    return body


# ---------------------------------------------------------------- helpers
def noise(p, scale=1.0, octaves=3, seed=0):
    """Cheap smooth 3D noise (sum of random plane waves), roughly in [-1, 1]."""
    r = np.random.default_rng(seed)
    v = np.zeros(len(p))
    for o in range(octaves):
        k = (0.08 * 2 ** o) / scale
        for _ in range(6):
            d = r.normal(size=3); d /= np.linalg.norm(d)
            v += np.sin(k * (p @ d) + r.uniform(0, 2 * np.pi)) / (2 ** o)
    return v / (6 * 1.2)


def spline_tube(pts, radius, n=200, scale_z=None, capping=True, sides=48):
    s = pv.Spline(np.asarray(pts, float), n)
    t = s.tube(radius=radius, n_sides=sides, capping=capping)
    if scale_z is not None:              # flatten (squashed bowel) about the local centreline height
        zc = np.interp(t.points[:, 1], s.points[:, 1], s.points[:, 2]) if np.ptp(s.points[:, 1]) > 1 else s.points[:, 2].mean()
        t.points[:, 2] = zc + (t.points[:, 2] - zc) * scale_z
    return t.triangulate().clean()


def arrow_along(pts, radius=0.45, head=2.6):
    """Curved arrow: tube along a spline + cone at the end."""
    s = pv.Spline(np.asarray(pts, float), 80)
    body = s.tube(radius=radius, n_sides=20)
    tip = s.points[-1]; d = s.points[-1] - s.points[-4]; d /= np.linalg.norm(d)
    cone = pv.Cone(center=tip + d * head * 0.5, direction=d, height=head, radius=radius * 2.6, resolution=30)
    return body.merge(cone.triangulate())


# ---------------------------------------------------------------- scene geometry
TOP = 4.835 + 0.40        # drain top skin (z) + ring height -> where tissue rests


def floor_z(y, back_cap=None):
    """Height of the organ bed: cradles the drain (wedge channel at each edge), rises gently to the
    sides and steeply behind (the cavity wall), so the far ends of the bowel loops disappear in it."""
    y = np.asarray(y, float); ay = np.abs(y)
    yb = y if back_cap is None else np.minimum(y, back_cap)
    side = np.where(y > 0, 0.22 * (ay - 13.0) + 0.02 * np.clip(yb - 50, 0, None) ** 2, 0.12 * (ay - 13.0))
    return np.where(ay < 13.0, -5.15 * np.sqrt(np.clip(1 - (ay / 13.0) ** 2, 0, 1)), side) - 0.3


def floor_surface():
    """Posterior wall / organ bed the drain lies on."""
    x = np.linspace(-230, 175, 406); y = np.linspace(-160, 112, 273)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ay = np.abs(yy)
    zz = floor_z(yy) + 0.9 * np.sin(0.05 * xx + 0.4) * (ay > 14) + 0.6 * np.cos(0.07 * yy) * (ay > 14)
    g = pv.StructuredGrid(xx, yy, zz).extract_surface(algorithm="dataset_surface").triangulate()
    g["s"] = noise(g.points, 1.4, seed=11)
    return g


def bowel_loops():
    """Bowel loops draped across the drain (squashed tubes), resting on the ring tops."""
    loops = []
    R, sq = 14.0, 0.70                     # radius, squash factor -> half height 9.8 mm
    zc = TOP + R * sq + 0.05               # centreline height where the loop crosses the drain
    for x0, tilt, seed in [(-40, 0.10, 1), (0.0, -0.06, 2), (40, 0.08, 3), (-82, 0.0, 4)]:
        ys = np.linspace(-80, 104, 19)
        xs = x0 + tilt * ys + 2.5 * np.sin(ys / 25 + seed)
        zfl = np.maximum(floor_z(ys, back_cap=55), 0)          # far ends run into the cavity wall
        zs = np.maximum(zc, zfl + R * sq + 0.3) if x0 > -70 else np.maximum(zfl + R * sq + 4.0, 12.0)
        loops.append(spline_tube(np.c_[xs, ys, zs], R, scale_z=sq))
    # one loop lying alongside the drain, behind it
    xs = np.linspace(-120, 100, 15); ys = 32 + 4 * np.sin(xs / 30); zs = floor_z(ys) + 10.7 + 0 * xs
    loops.append(spline_tube(np.c_[xs, ys, zs], 13.0, scale_z=0.8))
    out = []
    for i, L in enumerate(loops):
        L["s"] = noise(L.points, 1.0, seed=20 + i)
        out.append(L)
    return out


def catheter():
    """Irrigation catheter: comes down between two bowel loops; open tip ~15 mm above the drain."""
    tip = np.array([20.0, -1.0, TOP + 15.0])
    pts = [tip, tip + [0, 0.5, 12], tip + [1, 3, 30], tip + [6, 10, 55], tip + [14, 18, 85]]
    c = spline_tube(pts[::-1], 2.0, capping=False, sides=36)
    lumen = spline_tube(pts[::-1], 1.25, capping=False, sides=36)
    ring = pv.Disc(center=tip, inner=1.25, outer=2.0, normal=(0, 0, 1), c_res=36)
    return c, lumen, ring, tip


def outlet_tube():
    """Drain outlet tube leaving the body towards the skin exit and the suction."""
    pts = [[-80.5, 0, 0], [-95, 0, 0.5], [-108, 2, 8], [-116, 4, 26], [-121, 6, 70], [-123, 7, 140]]
    return spline_tube(pts, 2.8, capping=False, sides=36)


def liquid(tip):
    """Irrigation liquid: stream from the catheter tip onto the drain, a thin film spreading over the
    top face, and the wedge channels along both drain edges."""
    parts = []
    # falling stream (narrows as it falls)
    z = np.linspace(tip[2] - 0.3, TOP + 0.2, 40)
    s = pv.Spline(np.c_[tip[0] + 0.15 * (tip[2] - z), tip[1] + 0 * z, z], 40)
    s["r"] = np.linspace(1.2, 0.7, s.n_points)
    parts.append(s.tube(scalars="r", absolute=True, n_sides=24))
    # puddle where it lands (between the two loops) + film over the uncovered front half
    th = np.linspace(0, 2 * np.pi, 80)
    for cx, rx, ry in [(21.5, 7.5, 5.5)]:
        rr = 1 + 0.12 * np.sin(3 * th) + 0.08 * np.cos(5 * th)
        ring = np.c_[cx + rx * rr * np.cos(th), -1 + ry * rr * np.sin(th)]
    film_pts = []
    for (px, py) in ring:
        film_pts.append([px, py, top_z(py) + 0.55])
    poly = pv.PolyData(np.array(film_pts)).delaunay_2d()
    parts.append(poly.extrude((0, 0, 0.35), capping=True).triangulate())
    # thin film strip spreading along the front half of the top face, towards both ends
    xs = np.linspace(-58, 58, 233); ys = np.linspace(-8.2, -1.5, 16)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zz = top_z(yy) + 0.25
    sheet = pv.StructuredGrid(xx, yy, zz).extract_surface(algorithm="dataset_surface").triangulate()
    parts.append(sheet)
    # wedge channels along both edges (liquid between drain edge, floor and loops)
    for sgn in (-1, 1):
        xs = np.linspace(-62, 60, 60)
        pts = np.c_[xs, sgn * (9.9 + 0.25 * np.sin(xs / 9)), -0.6 + 0 * xs]
        parts.append(pv.Spline(pts, 120).tube(radius=1.15, n_sides=24))
    return parts


def top_z(y):
    """Height of the drain's top skin across the elliptical section (17.85 x 9.67 mm)."""
    return 4.835 * np.sqrt(np.clip(1 - (np.asarray(y) / 8.93) ** 2, 0, 1)) + 0.0


# ---------------------------------------------------------------- render
def world_to_px(pl, p, H):
    c = vtk.vtkCoordinate(); c.SetCoordinateSystemToWorld(); c.SetValue(*map(float, p))
    x, y = c.GetComputedDoubleDisplayValue(pl.renderer)
    return x, H - y


def main():
    drain = drain_body()
    floor = floor_surface()
    loops = bowel_loops()
    cat, lumen, cat_ring, tip = catheter()
    tube = outlet_tube()
    wet = liquid(tip)

    W, H = 2000, 1250
    pl = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
    pl.set_background("white")
    pl.add_light(pv.Light(position=(60, -160, 260), focal_point=(0, 0, 0), intensity=0.85, light_type="scene light"))
    pl.add_light(pv.Light(position=(-200, 100, 120), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    pl.add_light(pv.Light(position=(0, -300, 20), focal_point=(0, 0, 0), intensity=0.25, light_type="scene light"))

    from matplotlib.colors import LinearSegmentedColormap as LSC
    tissue_dark = LSC.from_list("bed", ["#8a3a45", "#b4606b", "#d58e96"])
    bowel = LSC.from_list("bowel", ["#c27078", "#dd9a9c", "#efbfb8"])
    pl.add_mesh(floor, scalars="s", cmap=tissue_dark, show_scalar_bar=False, smooth_shading=True, specular=0.15, ambient=0.3, diffuse=0.75)
    for L in loops:
        cut = L.clip_closed_surface(normal=(0, 1, 0), origin=(0, -1.0, 0)) if L.bounds[2] < -1 else L   # remove the front half
        cut = cut.compute_normals(split_vertices=True, feature_angle=35)   # crisp edge at the cut face
        cut["s"] = noise(cut.points, 1.0, seed=int(abs(cut.center[0])) + 5)
        pl.add_mesh(cut, scalars="s", cmap=bowel, show_scalar_bar=False, smooth_shading=True, specular=0.3,
                    specular_power=25, ambient=0.3, diffuse=0.75)
    pl.add_mesh(drain, color=GREY, smooth_shading=True, specular=0.4, specular_power=30, ambient=0.3, diffuse=0.8)
    pl.add_mesh(tube, color="#c9d3df", smooth_shading=True, specular=0.4, ambient=0.35)
    pl.add_mesh(cat, color=SILICONE, smooth_shading=True, specular=0.5, ambient=0.35)
    pl.add_mesh(lumen, color="#9a9483", smooth_shading=True)
    pl.add_mesh(cat_ring, color=SILICONE)
    for w in wet:
        pl.add_mesh(w, color=BLUE, opacity=0.55, smooth_shading=True, specular=0.8, specular_power=60)
    # flow arrows: liquid spreading along the top face (front half) and suction in the outlet tube
    arrows = [arrow_along([[16, -5, TOP + 2.2], [2, -5.5, TOP + 2.0], [-22, -5.5, TOP + 1.8]]),
              arrow_along([[27, -5, TOP + 2.2], [38, -5.5, TOP + 2.0], [56, -5.5, TOP + 1.8]]),
              arrow_along([[18, -9.5, 6.0], [16, -11.5, 2.5], [10, -11.2, 0.8]], radius=0.4)]
    for a in arrows:
        pl.add_mesh(a, color="#1d5fae", smooth_shading=True)
    pl.add_mesh(arrow_along([[-99, -7.5, 4.0], [-109, -5.5, 13], [-114, -3.5, 29], [-117, -2, 46]], radius=0.75, head=4.5),
                color=ORANGE, smooth_shading=True, ambient=0.3)

    pl.enable_depth_peeling(number_of_peels=8)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(55, -205, 165), (-22, 5, 10), (0, 0, 1)]
    pl.camera.view_angle = 30
    pl.camera.zoom(1.12)
    pl.render()

    anchors = {
        "drain": (-40, -8.5, 3.5), "rings": (48, -6.5, 4.6), "dome": (63, 0, 2),
        "cat": tip + np.array([2, 0, 10]), "stream": (20.6, -1, TOP + 7), "film": (-30, -6, TOP + 0.6),
        "edge": (-55, -10.4, -0.6), "loop": (-40, -1.2, TOP + 12), "floor": (-10, -45, -4),
        "tube": (-118, 6, 40),
    }
    px = {k: world_to_px(pl, v, H) for k, v in anchors.items()}
    raw = os.path.join(HERE, "in_body_3d_raw.png")
    pl.screenshot(raw)
    pl.close()
    json.dump(px, open(os.path.join(HERE, "_anchors.json"), "w"), indent=1)
    annotate(raw, px, W, H)


def annotate(raw, px, W, H):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(raw)
    fig = plt.figure(figsize=(W / 100, H / 100 + 1.3), dpi=100)
    ax = fig.add_axes([0, 0, 1, H / (H + 130)])
    ax.imshow(img); ax.set_axis_off()
    kw = dict(fontsize=17, ha="left", va="center", color="#1b2733",
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9aa7b4", lw=0.8, alpha=0.93),
              arrowprops=dict(arrowstyle="-", color="#33414f", lw=1.1, shrinkA=0, shrinkB=2))
    lab = [
        ("cat", "irrigation catheter from the bag\n(tip ~15 mm above the drain)", (0.62, 0.08)),
        ("stream", "irrigation liquid falls\nbetween two bowel loops", (0.72, 0.30)),
        ("rings", "slotted rings hold tissue\n0.4 mm off every hole", (0.80, 0.50)),
        ("dome", "closed dome tip", (0.86, 0.64)),
        ("loop", "bowel loops resting on the\nring tops (cut open here)", (0.27, 0.05)),
        ("tube", "outlet tube to skin exit\nand suction (-7 mmHg)", (0.13, 0.22)),
        ("drain", "drain: real CAD\n18 x 10 mm, ~120 mm long", (0.02, 0.60)),
        ("edge", "liquid channel along\neach drain edge", (0.07, 0.80)),
        ("film", "thin liquid film on the drain,\nunder the bowel, into the holes", (0.20, 0.92)),
        ("floor", "posterior wall / organ bed", (0.62, 0.93)),
    ]
    for k, text, (fx, fy) in lab:
        x, y = px[k]
        ax.annotate(text, xy=(x, y), xytext=(fx * W, fy * H), **kw)
    fig.text(0.5, 1 - 0.32 * 130 / (H + 130),
             "Drain inside the body with irrigation: blue = irrigation liquid, orange = suction direction",
             ha="center", va="center", fontsize=21, color="#1b2733")
    fig.text(0.5, 1 - 0.74 * 130 / (H + 130),
             "Drain = your standoff CAD. Organs, catheter and liquid are illustrative shapes; the gaps and the film are drawn thicker than real.",
             ha="center", va="center", fontsize=14, color="#55626e")
    out = os.path.join(HERE, "in_body_3d.png")
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--labels":
        px = json.load(open(os.path.join(HERE, "_anchors.json")))
        annotate(os.path.join(HERE, "in_body_3d_raw.png"), px, 2000, 1250)
    else:
        main()

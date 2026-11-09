"""
Realistic packing: the drain fully buried in tissue. Bowel loops lie across it, touching each
other; a second layer of loops lies on top, the omentum (fatty apron) drapes over everything, and
loops also lie along both drain edges. All tissue is drawn see-through so the drain, the rings,
the irrigation catheter and the liquid paths stay visible.

    py -3.12 make_in_body_3d_packed.py      ->  in_body_3d_packed.png

Uses the shapes and helpers of make_in_body_3d.py (same drain CAD, catheter, liquid, frame).
"""
import json, os
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap as LSC

import make_in_body_3d as base
from make_in_body_3d import (HERE, TOP, GREY, BLUE, SILICONE, ORANGE, drain_body, floor_surface, floor_z,
                             spline_tube, noise, catheter, outlet_tube, liquid, arrow_along, world_to_px)

R, SQ = 14.0, 0.70            # bowel radius, squash -> 9.8 mm half height
HALF = R * SQ


def crossing_loops():
    """Loops lying across the drain, side by side, resting on the ring tops. A narrow cleft is
    left at x ~ 20 where the irrigation catheter comes down."""
    out = []
    zc = TOP + HALF + 0.05
    for i, x0 in enumerate([-84, -56, -28, 4, 36, 66]):
        ys = np.linspace(-95, 104, 21)
        xs = x0 + 1.5 * np.sin(ys / 22 + i)
        zfl = np.maximum(floor_z(ys, back_cap=55), 0)
        zs = np.maximum(zc, zfl + HALF + 0.3)
        if x0 < -70:                                # over the tapered end / outlet tube
            zs = np.maximum(zfl + HALF + 4.0, 12.0)
        out.append(spline_tube(np.c_[xs, ys, zs], R, scale_z=SQ))
    return out


def edge_loops():
    """Loops lying along the drain, against both edges (they close the space beside it)."""
    out = []
    for y0 in (-25.5, 25.5):
        xs = np.linspace(-140, 110, 16)
        ys = y0 + 1.5 * np.sin(xs / 28)
        zs = floor_z(ys) + HALF + 0.4 + 0 * xs
        out.append(spline_tube(np.c_[xs, ys, zs], R, scale_z=SQ))
    return out


def upper_loops():
    """Second layer of loops on top of the crossing ones (gap at y ~ 0 for the catheter)."""
    out = []
    z = TOP + 2 * HALF + 9.6
    for y0, ph in ((-17.5, 0.0), (15.5, 1.3), (-45, 2.1), (44, 0.7)):
        xs = np.linspace(-150, 115, 16)
        ys = y0 + 2.0 * np.sin(xs / 30 + ph)
        zs = z + (0 if abs(y0) < 30 else 2.0) + 1.5 * np.sin(xs / 40 + ph)
        out.append(spline_tube(np.c_[xs, ys, zs], 13.0, scale_z=0.75))
    return out


def omentum():
    """Fatty apron draped over the bowel."""
    x = np.linspace(-150, 115, 133); y = np.linspace(-70, 70, 71)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = TOP + 2 * HALF + 22 + 2.5 * np.sin(xx / 17) * np.cos(yy / 13) - 0.004 * yy ** 2
    g = pv.StructuredGrid(xx, yy, zz).extract_surface(algorithm="dataset_surface").triangulate()
    g["s"] = noise(g.points, 1.2, seed=77)
    return g


def main():
    drain = drain_body()
    floor = floor_surface()
    cat, lumen, cat_ring, tip = catheter()
    tube = outlet_tube()
    wet = liquid(tip)

    W, H = 2000, 1250
    pl = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
    pl.set_background("white")
    pl.add_light(pv.Light(position=(60, -160, 260), focal_point=(0, 0, 0), intensity=0.85, light_type="scene light"))
    pl.add_light(pv.Light(position=(-200, 100, 120), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light"))
    pl.add_light(pv.Light(position=(0, -300, 20), focal_point=(0, 0, 0), intensity=0.25, light_type="scene light"))

    bed = LSC.from_list("bed", ["#8a3a45", "#b4606b", "#d58e96"])
    bowel = LSC.from_list("bowel", ["#c27078", "#dd9a9c", "#efbfb8"])
    fat = LSC.from_list("fat", ["#e6b85c", "#f2d27a", "#f8e3a6"])
    look = dict(show_scalar_bar=False, smooth_shading=True, specular=0.25, ambient=0.35, diffuse=0.7)

    pl.add_mesh(floor, scalars="s", cmap=bed, **look)
    for group, op in ((crossing_loops(), 0.33), (edge_loops(), 0.25), (upper_loops(), 0.22)):
        for i, L in enumerate(group):
            L["s"] = noise(L.points, 1.0, seed=40 + i)
            pl.add_mesh(L, scalars="s", cmap=bowel, opacity=op, **look)
    pl.add_mesh(omentum(), scalars="s", cmap=fat, opacity=0.18, **look)

    pl.add_mesh(drain, color="#5d7088", smooth_shading=True, specular=0.4, specular_power=30, ambient=0.35, diffuse=0.8)
    pl.add_mesh(tube, color="#c9d3df", smooth_shading=True, specular=0.4, ambient=0.35)
    pl.add_mesh(cat, color=SILICONE, smooth_shading=True, specular=0.5, ambient=0.35)
    pl.add_mesh(lumen, color="#9a9483", smooth_shading=True)
    pl.add_mesh(cat_ring, color=SILICONE)
    for i, w in enumerate(wet):          # wet[2] = film over the top face: lighter so the rings show
        pl.add_mesh(w, color=BLUE, opacity=0.4 if i == 2 else 0.85, smooth_shading=True, specular=0.8, specular_power=60)
    for a in [arrow_along([[16, -5, TOP + 2.2], [2, -5.5, TOP + 2.0], [-22, -5.5, TOP + 1.8]]),
              arrow_along([[27, -5, TOP + 2.2], [38, -5.5, TOP + 2.0], [56, -5.5, TOP + 1.8]])]:
        pl.add_mesh(a, color="#1d5fae", smooth_shading=True)
    pl.add_mesh(arrow_along([[-99, -7.5, 4.0], [-109, -5.5, 13], [-114, -3.5, 29], [-117, -2, 46]], radius=0.75, head=4.5),
                color=ORANGE, smooth_shading=True, ambient=0.3)

    pl.enable_depth_peeling(number_of_peels=16, occlusion_ratio=0.0)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(55, -205, 165), (-22, 5, 12), (0, 0, 1)]
    pl.camera.view_angle = 30
    pl.camera.zoom(1.08)
    pl.render()

    anchors = {
        "cross": (-28, -20, TOP + 15), "upper": (-60, -17.5, TOP + 30), "oment": (55, 25, TOP + 41),
        "edge": (-40, -25.5, 18), "drain": (-45, -8.5, 3.5), "cat": tip + np.array([2, 1, 25]),
        "stream": (20.6, -1, TOP + 7), "film": (-10, -6, TOP + 0.4), "dome": (63, 0, 2),
        "tube": (-118, 6, 40),
    }
    px = {k: world_to_px(pl, v, H) for k, v in anchors.items()}
    raw = os.path.join(HERE, "in_body_3d_packed_raw.png")
    pl.screenshot(raw)
    pl.close()
    annotate(raw, px, W, H)


def annotate(raw, px, W, H):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = plt.imread(raw)
    fig = plt.figure(figsize=(W / 100, H / 100 + 1.3), dpi=100)
    ax = fig.add_axes([0, 0, 1, H / (H + 130)])
    ax.imshow(img); ax.set_axis_off()
    kw = dict(fontsize=17, ha="left", va="center", color="#1b2733", annotation_clip=False,
              bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9aa7b4", lw=0.8, alpha=0.93),
              arrowprops=dict(arrowstyle="-", color="#33414f", lw=1.1, shrinkA=0, shrinkB=2))
    lab = [
        ("oment", "omentum (fatty apron)\ndraped over the bowel", (0.70, 0.06)),
        ("cat", "irrigation catheter, pushed in\nbetween the loops", (0.66, 0.24)),
        ("stream", "liquid opens its own thin path\ndown to the drain", (0.70, 0.42)),
        ("dome", "closed dome tip", (0.84, 0.60)),
        ("upper", "second layer of bowel loops", (0.20, 0.05)),
        ("cross", "loops lying across the drain,\ntouching each other and the rings", (0.02, 0.20)),
        ("tube", "outlet tube to suction", (0.02, 0.36)),
        ("drain", "drain (real CAD): covered on\nall sides, touching tissue everywhere", (0.02, 0.58)),
        ("edge", "loops along both drain edges", (0.05, 0.80)),
        ("film", "liquid only in thin films:\nunder the tissue, between rings, along the edges", (0.33, 0.93)),
    ]
    for k, text, (fx, fy) in lab:
        ax.annotate(text, xy=px[k], xytext=(fx * W, fy * H), **kw)
    fig.text(0.5, 1 - 0.32 * 130 / (H + 130),
             "Drain buried in tissue, tissue drawn see-through: blue = irrigation liquid, orange = suction",
             ha="center", va="center", fontsize=21, color="#1b2733")
    fig.text(0.5, 1 - 0.74 * 130 / (H + 130),
             "There is no empty space: organs touch the drain and each other. Liquid films (real: ~0.1-0.4 mm) are drawn thicker so they show.",
             ha="center", va="center", fontsize=14, color="#55626e")
    out = os.path.join(HERE, "in_body_3d_packed.png")
    fig.savefig(out, dpi=100)
    print("wrote", out)


if __name__ == "__main__":
    main()

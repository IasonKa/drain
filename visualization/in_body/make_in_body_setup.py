"""
Schematic of the whole set-up: patient lying on the back, the drain in the lowest part of the
abdominal cavity, an irrigation catheter whose tip sits a few cm above it, the irrigation bag and
pump, and the suction line to the canister and the wall regulator at -7 mmHg.
Two close-ups: a cut across the drain among the organs, and one ringed hole.

    py -3.12 make_in_body_setup.py      ->  in_body_setup.png

Illustrative only: the anatomy is simplified and the placement is an example, not clinical advice.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

SKIN, SKIN_E = "#f3d9c4", "#b98d6c"
CAV = "#f7e1e1"
ORG, ORG_E = "#e3a0a6", "#a4505b"
LIVER, LIVER_E = "#b9636b", "#7a2f38"
DRAIN, DRAIN_E = "#9fb0c3", "#4a5b6e"
BLUE, BLUE_D = "#4a90d9", "#1d5fae"
SIL = "#efe9d6"
ORANGE = "#e07a1f"
INK = "#1b2733"
MUTED = "#5a6774"


def catmull(pts, n=30, closed=False):
    """Smooth curve through points (Catmull-Rom)."""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return np.array(out)


def tube_line(ax, pts, color, lw, z=5, edge="#6b6455", ls="-"):
    c = catmull(pts, 25)
    ax.plot(c[:, 0], c[:, 1], color=edge, lw=lw + 1.6, solid_capstyle="round", zorder=z, ls=ls)
    ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, solid_capstyle="round", zorder=z + 0.1, ls=ls)
    return c


def label(ax, xy, text, xytext, fs=11.5, ha="left", z=30):
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, ha=ha, va="center", color=INK, zorder=z,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#a9b4bf", lw=0.7, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#3a4754", lw=0.9, shrinkA=0, shrinkB=1.5))


def flow_arrow(ax, p, q, color=BLUE_D, lw=1.8, ms=13, z=12, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


# ====================================================================== main panel: the set-up
def setup_panel(ax):
    ax.set_xlim(0, 150); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.axis("off")

    # wall, floor, bed

    ax.plot([0, 150], [3, 3], color="#c3cbd3", lw=1.2, zorder=0)
    ax.add_patch(FancyBboxPatch((8, 15.5), 118, 4.5, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#d8dee5", ec="#8996a3", lw=1, zorder=1))
    for x in (14, 120):
        ax.plot([x, x], [3, 15.5], color="#8996a3", lw=2.2, zorder=1)

    # patient, lying on the back, head left (side view)
    body = catmull([(20, 20.3), (20, 27), (24, 33), (31, 37), (40, 41.5), (52, 44), (66, 45.6), (80, 45),
                    (92, 41.5), (100, 36.5), (112, 31.5), (124, 28.5), (126, 24), (124, 20.3)], 18)
    ax.add_patch(Polygon(np.vstack([body, [[20, 20.3]]]), closed=True, fc=SKIN, ec=SKIN_E, lw=1.3, zorder=2))
    ax.add_patch(Circle((13.5, 28), 7.3, fc=SKIN, ec=SKIN_E, lw=1.3, zorder=2))
    ax.add_patch(Rectangle((5, 20.3), 13, 3.2, fc="white", ec="#b8c0c8", lw=0.8, zorder=1.5))   # pillow

    # abdominal cavity (cut open view)
    cav = catmull([(47, 22.6), (47.5, 33), (51, 40.5), (64, 42.8), (80, 42.3), (90, 39), (95, 33),
                   (95, 26), (92, 22.6)], 18)
    ax.add_patch(Polygon(np.vstack([cav, [[47, 22.6]]]), closed=True, fc=CAV, ec=ORG_E, lw=1.0, ls=(0, (4, 2)), zorder=3))
    # liver under the diaphragm
    liv = catmull([(48.5, 30), (49.5, 38.5), (54, 41.6), (62, 41.6), (63.5, 38), (59, 34.5), (53, 31.5)], 14, closed=True)
    ax.add_patch(Polygon(liv, closed=True, fc=LIVER, ec=LIVER_E, lw=1, zorder=4))
    # bowel loops (leave a cleft where the irrigation catheter comes down)
    rng = np.random.default_rng(4)
    loops = [(57, 27.0, 3.0), (60.5, 33.0, 3.2), (66.5, 38.6, 3.3), (64.5, 28.4, 3.0), (67.0, 33.4, 2.6),
             (78.0, 28.0, 3.0), (79.5, 33.8, 3.0), (78.0, 39.3, 2.9), (84.5, 30.0, 3.0), (85.5, 36.4, 3.0),
             (90.5, 27.2, 2.8), (90.6, 33.0, 2.6), (72.4, 39.8, 2.4), (54.3, 25.2, 2.2), (87.0, 24.7, 2.0)]
    for (x, y, r) in loops:
        ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * 0.86, angle=rng.uniform(0, 180), fc=ORG, ec=ORG_E, lw=0.9, zorder=4))
        ax.add_patch(Ellipse((x, y), r * 0.9, r * 0.6, angle=rng.uniform(0, 180), fc="none", ec=ORG_E, lw=0.6, zorder=4.1))

    # the drain in the lowest (posterior) part of the cavity, under the bowel
    d0, d1, dy = 59.5, 82.0, 23.0
    ax.add_patch(FancyBboxPatch((d0, dy), d1 - d0, 1.55, boxstyle="round,pad=0,rounding_size=0.75", fc=DRAIN, ec=DRAIN_E, lw=1, zorder=6))
    for x in np.arange(d0 + 1.2, d1 - 0.6, 1.05):
        ax.add_patch(Circle((x, dy + 0.78), 0.22, fc="#55667a", ec="none", zorder=6.1))
    # liquid: stream from the catheter tip down the cleft, film on the drain, pool around it
    tip = (72.0, 29.2)
    ax.add_patch(Polygon([(71.5, 29.0), (72.5, 29.0), (72.3, 24.7), (71.7, 24.7)], fc=BLUE, ec="none", alpha=0.8, zorder=5))
    ax.plot([d0 + 0.5, d1 - 0.5], [dy + 1.75, dy + 1.75], color=BLUE, lw=2.2, alpha=0.85, zorder=5.5, solid_capstyle="round")
    flow_arrow(ax, (70.3, 25.6), (64, 25.6), ms=10, lw=1.4)
    flow_arrow(ax, (73.7, 25.6), (80, 25.6), ms=10, lw=1.4)

    # drain outlet tube: from the drain end out through the flank skin, down to the canister
    tube_line(ax, [(d1, dy + 0.8), (87, 23.6), (95, 25.5), (101, 30.5), (104, 34.6)], "#c9d3df", 2.4, z=6.5, edge=DRAIN_E)
    ax.add_patch(Circle((104, 34.6), 0.9, fc="none", ec="#7a5a44", lw=1.2, zorder=7))     # skin exit
    tube_line(ax, [(104, 34.6), (107.5, 35.0), (110, 29), (108.5, 17), (107, 11.2)], "#c9d3df", 2.4, z=7, edge=DRAIN_E)
    # canister + regulator
    ax.add_patch(FancyBboxPatch((102.5, 3.6), 9, 7.4, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#f4f7fa", ec="#6d7b88", lw=1.2, zorder=7))
    ax.add_patch(Rectangle((102.8, 3.9), 8.4, 2.6, fc=BLUE, ec="none", alpha=0.55, zorder=7.1))
    for yy in np.arange(4.8, 10.5, 1.2):
        ax.plot([109.6, 111], [yy, yy], color="#6d7b88", lw=0.6, zorder=7.2)
    tube_line(ax, [(110, 11.2), (113, 16), (125, 22.5), (136, 23), (141.5, 23)], "#dfe5ea", 1.4, z=7, edge="#6d7b88")
    ax.add_patch(FancyBboxPatch((136.5, 21), 6.3, 9.2, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#e9edf1", ec="#55626e", lw=1.2, zorder=8))
    ax.add_patch(Circle((139.65, 27.1), 2.15, fc="white", ec="#55626e", lw=1.1, zorder=8.1))
    ax.plot([139.65, 138.6], [27.1, 28.6], color="#c0392b", lw=1.3, zorder=8.2)
    ax.text(139.65, 22.7, "-7", ha="center", va="center", fontsize=9, color=INK, zorder=8.3)
    flow_arrow(ax, (112.0, 33.0), (112.2, 25.5), color=ORANGE, ms=12, rad=-0.1)

    # irrigation: bag on a pole, drip chamber, pump, line into the abdomen
    ax.plot([34, 34], [3, 92], color="#7e8a96", lw=2.4, zorder=1)
    ax.plot([30, 38], [92, 92], color="#7e8a96", lw=2.0, zorder=1)
    ax.plot([29, 30, 31], [85.5, 92, 85.5], color="#7e8a96", lw=1.0, zorder=1)
    bag = catmull([(25.5, 86), (25, 78), (26.5, 72.5), (30, 71), (33.5, 72.5), (35, 78), (34.5, 86)], 14)
    ax.add_patch(Polygon(np.vstack([bag, [[25.5, 86]]]), closed=True, fc="#e6f0fb", ec="#5b7fa6", lw=1.2, zorder=2))
    ax.add_patch(Polygon(np.vstack([bag[bag[:, 1] < 80.5], [[34.6, 80.5], [25.4, 80.5]]]), closed=True, fc=BLUE, ec="none", alpha=0.35, zorder=2.1))
    ax.text(30, 82.7, "saline", ha="center", va="center", fontsize=9.5, color="#2c4a6b", zorder=3)
    ax.add_patch(FancyBboxPatch((28.9, 64.5), 2.2, 4.5, boxstyle="round,pad=0.1,rounding_size=0.6", fc="#f4f8fc", ec="#5b7fa6", lw=1, zorder=3))
    ax.plot([30, 30], [69, 71], color="#5b7fa6", lw=1.4, zorder=3)
    ax.add_patch(Circle((30, 66.0), 0.45, fc=BLUE, ec="none", zorder=3.1))
    ax.add_patch(FancyBboxPatch((35.2, 54), 9, 6.5, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#e9edf1", ec="#55626e", lw=1.2, zorder=3))
    ax.add_patch(Rectangle((36.2, 57.3), 4.2, 2.3, fc="#bfe3c8", ec="#55626e", lw=0.6, zorder=3.1))
    tube_line(ax, [(30, 64.5), (30.5, 60), (35.3, 57.2)], SIL, 1.6, z=3.2, edge="#8a8270")
    tube_line(ax, [(44.2, 57.2), (55, 56), (66, 52), (71.6, 47.2), (72, 45.4)], SIL, 1.8, z=3.2, edge="#8a8270")
    ax.add_patch(Circle((72, 45.4), 0.85, fc="none", ec="#7a5a44", lw=1.2, zorder=7))     # skin exit
    tube_line(ax, [(72, 45.4), (72.2, 42.0), (71.9, 36), (72.0, 31.5), tip], SIL, 1.8, z=6.8, edge="#8a8270")
    flow_arrow(ax, (58, 56.9), (64.5, 53.3), ms=12, rad=-0.1)

    # labels
    label(ax, (30, 78), "irrigation bag", (6, 95))
    label(ax, (39.7, 55.5), "pump: sets\nthe inflow rate", (4, 61))
    label(ax, (72.2, 45.6), "skin exit of the\nirrigation catheter", (74, 65))
    label(ax, tip, "catheter tip, a few cm\nabove the drain", (96, 57))
    label(ax, (62, 23.6), "drain: lowest part of the\ncavity, under the bowel", (40, 8.5))
    label(ax, (66.5, 41.6), "bowel loops", (51, 48.8))
    label(ax, (54, 38.5), "liver", (40, 45.5))
    label(ax, (104.3, 35.2), "skin exit of the drain", (108, 44))
    label(ax, (107, 7), "collection canister", (114, 13.5))
    label(ax, (139.6, 30.3), "wall suction\nregulator, -7 mmHg", (126, 37.5))
    ax.text(75, 98, "The set-up (side view, patient on the back)", fontsize=15, ha="center", va="center", color=INK)
    ax.text(75, 94.4, "blue = irrigation liquid, orange = suction; illustrative, not to scale", fontsize=10.5,
            ha="center", va="center", color=MUTED)


# ====================================================================== close-up 1: cut across the drain
def ellipse_pts(a, b, n=200, t0=0, t1=2 * np.pi):
    t = np.linspace(t0, t1, n)
    return np.c_[a * np.cos(t), b * np.sin(t)]


def wall_with_holes(ax, a, b, w, holes_x, ring_h=0.4, ring_r=(0.55, 0.85), z=10):
    """Elliptical drain wall (outer a x b, thickness w) with straight holes and ring arcs (section)."""
    ax.add_patch(Polygon(ellipse_pts(a, b), closed=True, fc=DRAIN, ec=DRAIN_E, lw=1, zorder=z))
    ax.add_patch(Polygon(ellipse_pts(a - w, b - w), closed=True, fc="#eef3f8", ec=DRAIN_E, lw=0.8, zorder=z + 0.1))
    for sgn in (1, -1):
        for hx in holes_x:
            yo = sgn * b * np.sqrt(1 - (hx / a) ** 2)
            yi = sgn * (b - w) * np.sqrt(max(1 - (hx / (a - w)) ** 2, 0))
            ax.add_patch(Rectangle((hx - 0.5, min(yo, yi) - 0.05), 1.0, abs(yo - yi) + 0.1, fc="#eef3f8", ec="none", zorder=z + 0.2))
            for side in (-1, 1):
                x0 = hx + side * ring_r[0]; x1 = hx + side * ring_r[1]
                xs = np.array([x0, x1]); ys = sgn * b * np.sqrt(np.clip(1 - (xs / a) ** 2, 0, 1))
                poly = [(x0, ys[0] - sgn * 0.05), (x1, ys[1] - sgn * 0.05), (x1, ys[1] + sgn * ring_h), (x0, ys[0] + sgn * ring_h)]
                ax.add_patch(Polygon(poly, closed=True, fc=DRAIN, ec=DRAIN_E, lw=0.7, zorder=z + 0.3))


def cross_section_panel(ax):
    ax.set_xlim(-24, 24); ax.set_ylim(-13, 24.6); ax.set_aspect("equal"); ax.axis("off")
    a, b, w = 8.93, 4.835, 0.6
    top = b + 0.4
    # posterior wall / organ bed, cradling the drain, with wedge gaps at the edges
    xs = np.linspace(-24, 24, 300)
    yb = np.where(np.abs(xs) < 10.6, -b * 1.02 * np.sqrt(np.clip(1 - (xs / 10.6) ** 2, 0, 1)) - 0.45, 0.16 * (np.abs(xs) - 10.6) - 0.45)
    ax.add_patch(Polygon(np.vstack([np.c_[xs, yb], [[24, -13], [-24, -13]]]), closed=True, fc=LIVER, ec=LIVER_E, lw=1, zorder=2, alpha=0.85))
    # wedge channels at the drain edges (liquid)
    for s in (-1, 1):
        ax.add_patch(Polygon([(s * 8.6, 1.2), (s * 8.95, -0.2), (s * 10.5, -0.4), (s * 11.6, 0.9)], closed=True, fc=BLUE, ec="none", alpha=0.75, zorder=3))
    # bowel loops on top, resting on the ring tops; cleft between them
    for cx, rx, ry in [(-10.0, 10.3, 6.4), (11.4, 10.0, 6.2)]:
        ax.add_patch(Ellipse((cx, top + ry + 0.12), 2 * rx, 2 * ry, fc=ORG, ec=ORG_E, lw=1, zorder=4))
        ax.add_patch(Ellipse((cx, top + ry + 0.12), rx * 0.9, ry * 0.75, fc="none", ec=ORG_E, lw=0.7, zorder=4.1))
    # liquid film between the drain top and the bowel, and the cleft
    film = ellipse_pts(a + 0.45, b + 0.45, 120, 0.06, np.pi - 0.06)
    ax.plot(film[:, 0], film[:, 1], color=BLUE, lw=2.0, alpha=0.9, zorder=3.5)
    ax.add_patch(Polygon([(-0.4, 14.5), (1.5, 14.5), (1.4, top + 0.5), (-0.3, top + 0.5)], closed=True, fc=BLUE, ec="none", alpha=0.75, zorder=3.6))
    # irrigation catheter tip above the cleft
    ax.add_patch(Rectangle((-1.2, 14.5), 3.2, 7.5, fc=SIL, ec="#8a8270", lw=1, zorder=5))
    ax.add_patch(Rectangle((-0.45, 14.5), 1.7, 7.5, fc="#cfc8b2", ec="none", zorder=5.1))
    # the drain itself (holes through both faces, ring arcs)
    wall_with_holes(ax, a, b, w, holes_x=[-4.6, 0.0, 4.6])
    # suction symbol (flow into the page, towards the outlet)
    ax.add_patch(Circle((0, 0), 1.3, fc="white", ec=ORANGE, lw=1.6, zorder=12))
    ax.plot([-0.9, 0.9], [-0.9, 0.9], color=ORANGE, lw=1.6, zorder=12.1); ax.plot([-0.9, 0.9], [0.9, -0.9], color=ORANGE, lw=1.6, zorder=12.1)
    # flow arrows
    flow_arrow(ax, (0.2, top + 1.1), (-5.6, top - 0.15), ms=10, lw=1.3, rad=0.12, z=13)
    flow_arrow(ax, (0.9, top + 1.1), (6.3, top - 0.3), ms=10, lw=1.3, rad=-0.12, z=13)
    flow_arrow(ax, (-4.6, b + 0.2), (-4.6, b - 1.5), ms=9, lw=1.2, z=13)
    flow_arrow(ax, (4.6, b + 0.2), (4.6, b - 1.5), ms=9, lw=1.2, z=13)
    # dimensions
    ax.annotate("", xy=(-a, -8.2), xytext=(a, -8.2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9), zorder=14)
    ax.text(0, -9.4, "18 mm", ha="center", va="center", fontsize=10, color="white", zorder=14,
            bbox=dict(boxstyle="round,pad=0.15", fc=LIVER_E, ec="none", alpha=0.9))
    label(ax, (1.0, 19.5), "irrigation catheter", (5.0, 20.3), fs=10.5)
    label(ax, (0.5, 11.5), "liquid runs down\nthe cleft", (-23.5, 18.8), fs=10.5)
    label(ax, (-3.0, 5.7), "thin film under the\nbowel, on the rings", (-23.5, 14.5), fs=10.5)
    label(ax, (14, 11), "bowel loop", (13.0, 16.5), fs=10.5)
    label(ax, (10.6, 0.3), "edge channel", (13.5, 3.2), fs=10.5)
    label(ax, (1.0, -1.0), "drain lumen: suction,\nliquid flows to the outlet", (6.5, -4.6), fs=10.5)
    label(ax, (-14, -5), "posterior wall", (-23.5, -10.8), fs=10.5)
    ax.text(0, 23.4, "Close-up 1: cut across the drain", fontsize=13, ha="center", va="center", color=INK)


# ====================================================================== close-up 2: one ringed hole
def hole_panel(ax):
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-1.7, 1.75); ax.set_aspect("equal"); ax.axis("off")
    h = 0.4
    # lumen
    ax.add_patch(Rectangle((-2.6, -1.7), 5.2, 1.1, fc="#e3edf7", ec="none", zorder=1))
    # wall with the hole, ring arcs
    for x0, x1 in [(-2.6, -0.5), (0.5, 2.6)]:
        ax.add_patch(Rectangle((x0, -0.6), x1 - x0, 0.6, fc=DRAIN, ec=DRAIN_E, lw=1, zorder=5))
    for x0 in (-0.85, 0.55):
        ax.add_patch(Rectangle((x0, 0), 0.30, h, fc=DRAIN, ec=DRAIN_E, lw=1, zorder=5))
    for x0 in (-2.15, 1.85):       # neighbouring rings
        ax.add_patch(Rectangle((x0, 0), 0.30, h, fc=DRAIN, ec=DRAIN_E, lw=1, zorder=5))
    # slot behind the cut plane (dashed outline)
    for x0 in (-0.85, 0.55):
        ax.add_patch(Rectangle((x0 - 0.02, 0), 0.34, h, fc="none", ec=BLUE_D, lw=0.9, ls=(0, (2, 1.5)), zorder=6))
    # liquid under the tissue
    xs = np.linspace(-2.6, 2.6, 300)
    sag = h + 0.02 - 0.07 * np.exp(-(xs / 0.35) ** 2) - 0.05 * np.exp(-((np.abs(xs) - 1.3) / 0.28) ** 2)
    ax.add_patch(Polygon(np.vstack([np.c_[xs, sag], [[2.6, 0], [-2.6, 0]]]), closed=True, fc=BLUE, ec="none", alpha=0.45, zorder=3))
    ax.add_patch(Rectangle((-0.5, -0.6), 1.0, 0.62, fc=BLUE, ec="none", alpha=0.45, zorder=3))
    # tissue resting on the ring tops
    ax.add_patch(Polygon(np.vstack([np.c_[xs, sag], [[2.6, 1.75], [-2.6, 1.75]]]), closed=True, fc=ORG, ec=ORG_E, lw=1.1, zorder=4))
    # flows
    flow_arrow(ax, (-2.3, 0.17), (-1.05, 0.17), ms=11, lw=1.4, z=8)
    flow_arrow(ax, (2.3, 0.17), (1.05, 0.17), ms=11, lw=1.4, z=8)
    flow_arrow(ax, (0, 0.2), (0, -1.15), ms=12, lw=1.6, z=8)
    ax.text(0, -1.42, "into the drain (suction)", ha="center", va="center", fontsize=10.5, color=BLUE_D, zorder=9)
    # dims
    ax.annotate("", xy=(2.45, 0), xytext=(2.45, h), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9), zorder=9)
    ax.text(2.42, 0.62, "0.4 mm", ha="right", va="center", fontsize=10, color=INK, zorder=9)
    ax.annotate("", xy=(-0.5, -0.75), xytext=(0.5, -0.75), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9), zorder=9)
    ax.text(0.62, -0.75, "1 mm hole", ha="left", va="center", fontsize=10, color=INK, zorder=9)
    label(ax, (-1.6, 0.9), "organ surface rests on the ring tops", (-2.5, 1.45), fs=10.5)
    label(ax, (0.72, 0.3), "slot: side path\nfor the liquid", (1.0, 1.0), fs=10.5)
    label(ax, (-2.0, -0.3), "drain wall 0.6 mm", (-2.5, -1.0), fs=10.5)
    ax.text(0, 1.98, "Close-up 2: one hole with its slotted ring", fontsize=13, ha="center", va="center", color=INK)


def main():
    fig = plt.figure(figsize=(20, 9.8), dpi=110)
    setup_panel(fig.add_axes([0.0, 0.0, 0.60, 1.0]))
    cross_section_panel(fig.add_axes([0.605, 0.47, 0.39, 0.50]))
    hole_panel(fig.add_axes([0.605, 0.02, 0.39, 0.42]))
    fig.add_artist(plt.Line2D([0.60, 0.60], [0.04, 0.96], color="#d0d6dc", lw=1))
    out = os.path.join(HERE, "in_body_setup.png")
    fig.savefig(out, dpi=110, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()

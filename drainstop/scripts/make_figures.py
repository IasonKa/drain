"""Make all figures for the paper from results/*.json (PDF for LaTeX plus PNG previews).

    py -3.12 scripts/make_figures.py [name ...]
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches, colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from drainstop.model import (Drain, DrainModel, Ellipse, LAYOUTS, PATTERN_KEYS, MMHG,  # noqa: E402
                             rows_pattern, fixed_area_pattern, usz_as_built, segment_fraction)
import run_study  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.edgecolor": "#5f5e5a",
    "axes.labelcolor": "#2c2c2a", "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a",
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "pdf.fonttype": 42, "axes.titleweight": "normal",
})
W1 = 6.3                                        # text width (in)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#2c2c2a", "#898781", "#e1e0d9"
SKIN, SKIN_EDGE = "#e9e8e2", "#5f5e5a"
WATER, WATER_EDGE, AIR = "#cfe3f6", "#3987e5", "#e34948"
DESIGN_COLOR = {"recommended": CAT[0], "usz": CAT[1], "side2": CAT[2], "cross4": CAT[3]}
DESIGN_LABEL = {"recommended": "Recommended", "usz": "USZ as built", "side2": "Side rows",
                "cross4": "Four rows"}
DESIGN_LS = {"recommended": "-", "usz": "--", "side2": "-.", "cross4": ":"}
TILTS = [-10, -5, 0, 5, 10, 15, 20]
DRAIN = Drain()
ELL = Ellipse(DRAIN.a, DRAIN.b)


def tilt_color(t):
    """Diverging: blue for outlet down, gray at 0, red for outlet up."""
    if t == 0:
        return "#7a7974"
    if t < 0:
        return {-5: "#7fb0ea", -10: "#2a78d6", -15: "#1b4f91"}[t]
    return {5: "#f2a39a", 10: "#e66767", 15: "#c43b3a", 20: "#8f2524"}[t]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170)
    plt.close(fig)
    print("  ", name)


def recommended_spec():
    p = os.path.join(RES, "recommended.json")
    if os.path.exists(p):
        return load("recommended.json")
    return dict(layout="custom", angle=45.0, d=1.0, n_x=12, family="custom", tilt=0)


def design_pattern(name):
    if name == "recommended":
        return run_study.build(recommended_spec())
    if name == "usz":
        return usz_as_built()
    return fixed_area_pattern(name, 1.0, run_study.A_TARGET)


def hole_outline(x0, th0, d, n=40):
    """Opening boundary on the skin: returns x, y, z (mm) in the drain frame."""
    s0 = ELL.s_of(th0)
    ang = np.linspace(0, 2 * np.pi, n)
    s = s0 + 0.5 * d * np.cos(ang)
    x = x0 + 0.5 * d * np.sin(ang)
    th = ELL.theta(s)
    y, z = ELL.yz(th)
    return x, y, z


def dim(ax, p0, p1, text, off=(0, 0), fs=7.5, color=INK, text_off=(0, 0), ha="center", va="center"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", lw=0.7, color=color,
                                                      shrinkA=0, shrinkB=0))
    xm, ym = 0.5 * (p0[0] + p1[0]) + text_off[0], 0.5 * (p0[1] + p1[1]) + text_off[1]
    ax.text(xm, ym, text, fontsize=fs, color=color, ha=ha, va=va,
            bbox=dict(fc="white", ec="none", pad=0.6))


def clean(ax):
    ax.set_axis_off()
    ax.set_aspect("equal")


# ----------------------------------------------------------------------------- geometry
def side_outline(ax, alpha=0.0, fc=SKIN, lw=1.0, tube_to=145.0, zorder=2):
    """Drain side view (x-z) rotated by alpha (deg) about the bottom of the tip; returns the transform."""
    b, L = DRAIN.b, DRAIN.length
    zz = np.linspace(0, 2 * b, 60)
    cap = -b * np.sqrt(np.clip(1 - ((zz - b) / b) ** 2, 0, 1))      # dome tip, 4.8 mm long
    body = np.c_[np.r_[L, cap[::-1], L], np.r_[2 * b, zz[::-1], 0]]
    taper = np.array([[L, b - 4.0], [L + 6, b - 2.8], [tube_to, b - 2.8],
                      [tube_to, b + 2.8], [L + 6, b + 2.8], [L, b + 4.0]])
    ca, sa = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))

    def tr(P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.c_[P[:, 0] * ca - P[:, 1] * sa, P[:, 0] * sa + P[:, 1] * ca]
    ax.add_patch(patches.Polygon(tr(taper), closed=True, fc=fc, ec=SKIN_EDGE, lw=lw, zorder=zorder))
    ax.add_patch(patches.Polygon(tr(body), closed=True, fc=fc, ec=SKIN_EDGE, lw=lw, zorder=zorder + 0.1))
    return tr


def panel(fig, rect, xlim, ylim, title):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, loc="left", fontsize=8.5, pad=3)
    return ax


def near_side_holes(ax, pat, tr=None, zorder=4):
    for xi, th, d in zip(pat.x, pat.theta, pat.d):
        if math.sin(th) >= 0:                        # holes on the side facing the reader
            x, y, z = hole_outline(xi, th, d)
            P = np.c_[x, z] if tr is None else tr(np.c_[x, z])
            ax.add_patch(patches.Polygon(P, closed=True, fc="#2c2c2a", ec="none", zorder=zorder))


def fig_geometry():
    pat = design_pattern("recommended")
    b, L = DRAIN.b, DRAIN.length
    rp = DRAIN.port_d / 2
    fig = plt.figure(figsize=(W1, 5.0))

    ax = panel(fig, [0.01, 0.79, 0.98, 0.17], (-16, 176), (-7, 15),
               "(a) Side view, lying flat, to scale (recommended hole layout on the near side)")
    side_outline(ax)
    near_side_holes(ax, pat)
    ax.plot([-10, 150], [b, b], color=MUTED, lw=0.5, ls=(0, (8, 2, 1, 2)), zorder=1)
    dim(ax, (0, -3.6), (L, -3.6), "119 mm, closed tip to outlet end face")
    dim(ax, (DRAIN.x_lo, 12.6), (DRAIN.x_hi, 12.6), "perforated span, 5 to 113 mm")
    dim(ax, (-9.0, 0), (-9.0, 2 * b), "9.67", text_off=(-3.6, 0))
    ax.text(150, b, r"to suction, $p_s=-7$ mmHg", fontsize=7.2, va="center", ha="left")

    ax = panel(fig, [0.01, 0.43, 0.53, 0.30], (96, 148), (-4.2, 13.5),
               "(b) Outlet end, enlarged: centred bore and its lip")
    side_outline(ax)
    near_side_holes(ax, pat)
    for zz in (b - rp, b + rp):
        ax.plot([L - 3, 147], [zz, zz], color=INK, lw=0.6, ls=(0, (3, 2)), zorder=5)
    ax.plot([96, 147], [b, b], color=MUTED, lw=0.5, ls=(0, (8, 2, 1, 2)), zorder=1)
    ax.plot([L - 6, 140], [b - rp, b - rp], color=CAT[1], lw=0.8, zorder=6)
    ax.plot([L - 6, 140], [0, 0], color=CAT[1], lw=0.5, zorder=6)
    dim(ax, (137, 0), (137, b - rp), r"lip $z_\ell$ = 2.72", color=CAT[1], text_off=(5.8, -2.4))
    dim(ax, (130, b - rp), (130, b + rp), "Ø4.24 bore", text_off=(0, 3.9))
    dim(ax, (98, 0), (98, 2 * b), "9.67", text_off=(-2.6, 0))
    ax.text(103, 11.2, "end face, x = 119", fontsize=6.8, color=MUTED)
    ax.annotate("", xy=(L, 9.6), xytext=(112, 11.0), arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED))
    ax.text(124, 9.4, "tube Ø5.6", fontsize=6.8, color=MUTED, ha="center")

    ax = panel(fig, [0.56, 0.43, 0.43, 0.30], (-15, 17), (-4.5, 14.5),
               r"(c) End view: angle $\theta$ around the skin")
    th = np.linspace(-np.pi, np.pi, 300)
    y, z = ELL.yz(th)
    ax.fill(y, z, fc=SKIN, ec=SKIN_EDGE, lw=1.0)
    ai, bi = DRAIN.a - DRAIN.wall, DRAIN.b - DRAIN.wall
    ax.fill(ai * np.sin(th), b - bi * np.cos(th), fc="white", ec=SKIN_EDGE, lw=0.7)
    ax.add_patch(patches.Circle((0, b), 2.8, fc="none", ec=SKIN_EDGE, lw=0.7, ls="--"))
    ax.add_patch(patches.Circle((0, b), rp, fc="#d3d1c7", ec=INK, lw=0.9))
    ax.plot([-6.2, 6.2], [b - rp, b - rp], color=CAT[1], lw=0.8)
    for ang in (0, 30, -30, 90, 180):
        t = math.radians(ang)
        yy, zz2 = ELL.yz(t)
        ax.plot([0, yy], [b, zz2], color=MUTED, lw=0.5, ls=":")
        ax.plot(yy, zz2, "o", ms=3.2, color=CAT[0] if abs(ang) == 30 else INK, zorder=5)
    ax.text(0, -1.5, "0°, bottom (6 o'clock)", fontsize=6.6, ha="center", va="center")
    ax.text(9.4, b, "90°, edge\n(3 o'clock)", fontsize=6.6, ha="left", va="center")
    ax.text(0.7, 10.5, "180°, top (12 o'clock)", fontsize=6.6, ha="left", va="center")
    ax.text(7.4, 0.3, "+30°", fontsize=6.6, color=CAT[0], ha="left", va="center")
    ax.text(-7.4, 0.3, "−30°", fontsize=6.6, color=CAT[0], ha="right", va="center")
    dim(ax, (-DRAIN.a, 12.6), (DRAIN.a, 12.6), "17.85 mm")
    dim(ax, (-11.5, 0), (-11.5, 2 * b), "9.67", text_off=(-2.4, 0))
    dim(ax, (-3.0, 0.6), (-3.0, b - rp), "2.72", color=CAT[1], text_off=(-1.8, 0.2), fs=6.6)
    ax.annotate("bore Ø4.24\n(tube Ø5.6)", xy=(1.9, b + 1.2), xytext=(10.0, 7.7), fontsize=6.6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED), va="center")

    ax = panel(fig, [0.01, 0.02, 0.98, 0.34], (-22, 172), (-6, 36),
               r"(d) Tilt $\alpha$ (outlet up $>0$), pool level $h$ and lip height $z_\ell(\alpha)$")
    al = 10.0
    tr = side_outline(ax, alpha=al, tube_to=133)
    H = 9.0
    ax.add_patch(patches.Rectangle((-20, -5), 185, H + 5, fc=WATER, ec="none", zorder=0))
    ax.plot([-20, 165], [H, H], color=WATER_EDGE, lw=0.9, zorder=1)
    ax.plot([-20, 165], [0, 0], color=MUTED, lw=0.5, ls="--", zorder=1)
    lip = tr([[L, b - rp]])[0]
    ax.plot([lip[0] - 2, 160], [lip[1], lip[1]], color=CAT[1], lw=0.8, zorder=6)
    dim(ax, (156, 0), (156, lip[1]), r"$z_\ell(\alpha)$", text_off=(8, 0), color=CAT[1])
    dim(ax, (-14, 0), (-14, H), r"$h$", text_off=(-3.2, 0), color=WATER_EDGE)
    ax.add_patch(patches.Arc((0, 0), 70, 70, theta1=0, theta2=al, color=INK, lw=0.7, zorder=7))
    ax.text(37, 2.6, r"$\alpha$", fontsize=8.5, zorder=7)
    ax.text(150, H + 1.4, "pool level", fontsize=7, color=WATER_EDGE, ha="right")
    ax.text(150, -3.3, "level of the lowest point of the skin (h = 0)", fontsize=6.8, color=MUTED, ha="right")
    save(fig, "fig01_geometry")


def fig_mechanism():
    b, L = DRAIN.b, DRAIN.length
    rp = DRAIN.port_d / 2
    fig = plt.figure(figsize=(W1, 4.3))
    ax = panel(fig, [0.01, 0.60, 0.98, 0.34], (-14, 190), (-7, 30),
               r"(a) Partly drained, tilted drain: air entering the exposed holes lowers the vacuum $p_g$ inside")
    al = 8.0
    tr = side_outline(ax, alpha=al, tube_to=140, fc="white", lw=1.1, zorder=3)
    H = 7.0
    ax.add_patch(patches.Rectangle((-12, -5), 180, H + 5, fc=WATER, ec="none", zorder=0))
    ax.plot([-12, 168], [H, H], color=WATER_EDGE, lw=0.9, zorder=1)
    lip = tr([[L, b - rp]])[0]
    zeta = lip[1] + 0.6
    inner = np.c_[np.r_[0.6, L, L, 0.6], np.r_[0.6, 0.6, 2 * b - 0.6, 2 * b - 0.6]]
    clip = patches.Polygon(tr(inner), closed=True, fc="none", ec="none", zorder=4)
    ax.add_patch(clip)
    wat = patches.Rectangle((-12, -5), 180, zeta + 5, fc="#8ec0ef", ec="none", alpha=0.55, zorder=4)
    ax.add_patch(wat)
    wat.set_clip_path(clip)
    ax.plot([-8, 168], [zeta, zeta], color="#1b4f91", lw=0.6, ls="--", zorder=5)
    for xi in np.linspace(14, 110, 9):
        for zl in (2 * b, 0.0):
            P = tr([[xi, zl]])[0]
            above = zl > 0
            sgn = 1 if above else -1
            col = AIR if P[1] > H else WATER_EDGE
            ax.annotate("", xy=(P[0], P[1] + 0.4 * sgn), xytext=(P[0], P[1] + 3.6 * sgn),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, mutation_scale=7), zorder=7)
    P = tr([[L + 22, b]])[0]
    ax.annotate("", xy=(P[0] + 13, P[1]), xytext=(P[0] - 2, P[1]),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=8), zorder=7)
    ax.text(P[0] + 14, P[1], r"to $p_s$", fontsize=7.5, va="center")
    ax.text(170, zeta, r"inner level $\zeta = h + |p_g|/\rho g$", fontsize=7, va="center", color="#1b4f91")
    ax.text(170, H, r"pool level $h$", fontsize=7, va="center", color=WATER_EDGE)
    ax.text(170, lip[1] - 2.4, r"bore lip $z_\ell$", fontsize=7, va="center", color=CAT[1])
    ax.plot([lip[0] - 1, 168], [lip[1], lip[1]], color=CAT[1], lw=0.8, zorder=8, ls=(0, (1, 1.5)))
    ax.text(18, 25.5, "exposed holes: air in", color=AIR, fontsize=7.5)
    ax.text(18, -5.6, "submerged holes: liquid in", color=WATER_EDGE, fontsize=7.5)

    ax = fig.add_axes([0.0, 0.02, 0.47, 0.50])
    ax.set_axis_off()
    ax.set_xlim(-4.2, 8.6)
    ax.set_ylim(-0.8, 5.1)
    ax.set_title("(b) Flow network at one pool level", loc="left", fontsize=8.5, pad=3)

    def box(xy, text, fc, w=3.6, h=0.95):
        ax.add_patch(patches.FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                            boxstyle="round,pad=0.05,rounding_size=0.18",
                                            fc=fc, ec=SKIN_EDGE, lw=0.7))
        ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=7.0)

    def arrow(p0, p1, color=INK):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                                          mutation_scale=8, shrinkA=1, shrinkB=1))
    box((0.4, 4.4), r"air, $p = 0$", "#fbe3e1", w=2.6)
    box((4.6, 4.4), r"pool, $\rho g (h - z)$", WATER, w=2.8)
    box((2.5, 2.25), r"drain interior: gas $p_g$, inner level $\zeta$", "#f1efe8", w=5.6)
    box((2.5, 0.0), r"suction, $p_s = -933$ Pa", "#e1e0d9", w=3.6)
    arrow((0.4, 3.92), (1.4, 2.73), AIR)
    arrow((4.6, 3.92), (3.6, 2.73), WATER_EDGE)
    arrow((2.5, 1.77), (2.5, 0.48))
    ax.text(0.65, 3.3, "exposed\nholes\n" + r"$\sum C_h A_e$", fontsize=6.6, color=AIR, ha="right", va="center")
    ax.text(4.4, 3.3, "submerged\nholes", fontsize=6.6, color=WATER_EDGE, ha="left", va="center")
    ax.text(2.7, 1.12, r"bore $C_p A_p$: gas above $\zeta$," + "\nliquid below", fontsize=6.6, ha="left", va="center")

    ax = fig.add_axes([0.6, 0.1, 0.38, 0.40])
    lam = np.logspace(-1, 2, 300)
    Hv = 933.3 / (1126 * 9.81) * 1e3
    ax.loglog(lam, Hv / (1 + lam ** 2), color=INK, lw=1.8)
    for t in (0, 5, 10, 15, 20):
        m = DrainModel(design_pattern("recommended"), t)
        zl = m.lip_height()
        ax.axhline(zl, color=tilt_color(t), lw=0.9, ls="--")
        lstar = math.sqrt(max(Hv / zl - 1, 0))
        ax.plot(lstar, zl, "o", color=tilt_color(t), ms=4)
        right = t in (5, 15)
        ax.text(90 if right else 0.11, zl * 1.1, rf"$z_\ell$, {t}°", fontsize=6.3, color=tilt_color(t),
                ha="right" if right else "left")
    ax.set_xlabel(r"$\Lambda = \sum C_h A_e / (C_p A_p)$")
    ax.set_ylabel(r"lift $H_v/(1+\Lambda^2)$ (mm)")
    ax.set_ylim(0.3, 300)
    ax.text(0.11, 130, r"$H_v = |p_s|/\rho g = 84.5$ mm", fontsize=6.8)
    ax.set_title("(c) Lift left by the pressure divider", loc="left", fontsize=8.5)
    save(fig, "fig02_mechanism")


# ----------------------------------------------------------------------------- patterns
def draw_map(ax, pat, title, show_y=True, show_x=True):
    P = ELL.perimeter
    ax.add_patch(patches.Rectangle((0, -P / 4), DRAIN.length, P / 2, fc="#eef3fb", ec="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, P / 4), DRAIN.length, P / 4, fc="#f6f5f1", ec="none", zorder=0))
    ax.add_patch(patches.Rectangle((0, -P / 2), DRAIN.length, P / 4, fc="#f6f5f1", ec="none", zorder=0))
    for s in (-P / 4, P / 4):
        ax.axhline(s, color=MUTED, lw=0.4, ls=":")
    s = ELL.s_of(pat.theta)
    for xi, si, d, sc in zip(pat.x, s, pat.d, pat.area_scale):
        ax.add_patch(patches.Circle((xi, si), 0.5 * d * math.sqrt(sc), fc="#2c2c2a", ec="none", zorder=2))
    ax.set_xlim(-1, DRAIN.length + 1)
    ax.set_ylim(-P / 2, P / 2)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_yticks([-P / 2, -P / 4, 0, P / 4, P / 2])
    ax.set_yticklabels(["top", "−edge", "bottom", "+edge", "top"] if show_y else [])
    ax.tick_params(length=2, labelsize=6.5)
    ax.set_xticks([0, 30, 60, 90, 119])
    ax.tick_params(labelbottom=show_x)
    ax.set_title(title, loc="left", fontsize=7.6, pad=2)


def end_icon(ax, pat):
    th = np.linspace(-np.pi, np.pi, 200)
    y, z = ELL.yz(th)
    ax.fill(y, z, fc=SKIN, ec=SKIN_EDGE, lw=0.6)
    for t in np.unique(np.round(pat.theta, 4)):
        yy, zz = ELL.yz(t)
        ax.plot(yy, zz, "o", ms=2.6, color="#2c2c2a")
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1, 10.7)


def fig_patterns():
    keys = ["usz"] + PATTERN_KEYS + ["recommended"]
    fig = plt.figure(figsize=(W1, 8.6))
    gs = fig.add_gridspec(6, 4, width_ratios=[1, 0.22, 1, 0.22], hspace=0.62, wspace=0.08)
    for i, k in enumerate(keys):
        r, c = i // 2, (i % 2) * 2
        if k == "usz":
            pat, title = usz_as_built(), "USZ as built: 318 × Ø1.0 through both faces"
        elif k == "recommended":
            pat = design_pattern("recommended")
            sp = recommended_spec()
            title = f"Recommended: ±{sp['angle']:g}° staggered, {pat.n} × Ø{sp['d']:g}"
        else:
            pat = rows_pattern(k, 1.0, run_study.N_X)
            title = f"{chr(65 + i - 1)}  {LAYOUTS[k][0]}"
        ax = fig.add_subplot(gs[r, c])
        draw_map(ax, pat, title, show_y=(c == 0), show_x=(r == 5))
        ax2 = fig.add_subplot(gs[r, c + 1])
        end_icon(ax2, pat)
    fig.text(0.5, 0.075, "position along the drain from the tip, x (mm)", ha="center", fontsize=8)
    fig.text(0.04, 0.5, "position around the skin (developed arc length)", rotation=90, va="center", fontsize=8)
    save(fig, "fig03_patterns")


def fig_sizes():
    fig, axes = plt.subplots(8, 2, figsize=(W1, 5.6))
    for i, d in enumerate(run_study.SIZES):
        for j, fam in enumerate(("fixed_count", "fixed_area")):
            ax = axes[i, j]
            pat = (rows_pattern("cross4", d, run_study.N_X) if fam == "fixed_count"
                   else fixed_area_pattern("cross4", d, run_study.A_TARGET))
            draw_map(ax, pat, "", show_y=False, show_x=(i == 7))
            ax.set_yticks([])
            ax.text(121, 0, f"Ø{d:g}: {pat.n} holes,\n{pat.open_area_mm2:.0f} mm²", fontsize=6.3, va="center")
    axes[0, 0].set_title("Fixed count: 19 stations per row", loc="left", fontsize=8)
    axes[0, 1].set_title("Fixed open area: about 2 × bore area (28 mm²)", loc="left", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("x (mm)")
    fig.subplots_adjust(hspace=0.15, wspace=0.32)
    save(fig, "fig04_sizes")


def fig_3d():
    names = ["usz", "side2", "cross4", "recommended"]
    fig = plt.figure(figsize=(W1, 3.3))
    elev, azim = -24, -58
    light = np.array([-0.35, -0.55, -0.76])
    light /= np.linalg.norm(light)
    P = ELL.perimeter
    nx, ns = 600, 240
    xe = np.linspace(0, DRAIN.length, nx)
    se = np.linspace(-P / 2, P / 2, ns)
    S, X = np.meshgrid(se, xe)
    Y, Z = ELL.yz(ELL.theta(S))
    xc, sc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (se[1:] + se[:-1])
    thc = ELL.theta(sc)
    nrm = np.stack([np.zeros_like(thc), np.sin(thc) / DRAIN.a, -np.cos(thc) / DRAIN.b], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    shade_s = 0.62 + 0.38 * np.clip(nrm @ light, 0, 1)                 # per arc position
    base = np.array(mcolors.to_rgb("#d9d7cf"))
    hole_rgb = np.array(mcolors.to_rgb("#2c2c2a"))
    for i, n in enumerate(names):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        pat = design_pattern(n)
        fc = np.empty((nx - 1, ns - 1, 4))
        fc[..., :3] = base[None, None, :] * shade_s[None, :, None]
        fc[..., 3] = 1.0
        s_h = ELL.s_of(pat.theta)
        for x0, s0, d in zip(pat.x, s_h, pat.d):
            r = 0.5 * d
            ix = np.where(np.abs(xc - x0) <= r)[0]
            for ss in (s0, s0 - P, s0 + P):
                js = np.where(np.abs(sc - ss) <= r)[0]
                if ix.size and js.size:
                    m = (xc[ix, None] - x0) ** 2 + (sc[None, js] - ss) ** 2 <= r * r
                    sub = fc[np.ix_(ix, js)]
                    sub[m, :3] = hole_rgb
                    fc[np.ix_(ix, js)] = sub
        ax.plot_surface(X, Y, Z, facecolors=fc, shade=False, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, rasterized=True)
        ang = np.linspace(0, 2 * np.pi, 40)
        Xt, At = np.meshgrid(np.linspace(DRAIN.length, DRAIN.length + 22, 4), ang)
        ax.plot_surface(Xt, 2.8 * np.cos(At), DRAIN.b + 2.8 * np.sin(At), color="#bdbbb3", lw=0,
                        shade=True, rasterized=True)
        ps_, th_ = np.meshgrid(np.linspace(0, np.pi / 2, 12), np.linspace(-np.pi, np.pi, 60))
        ax.plot_surface(-DRAIN.b * np.sin(ps_), DRAIN.a * np.cos(ps_) * np.sin(th_),
                        DRAIN.b - DRAIN.b * np.cos(ps_) * np.cos(th_), color="#d9d7cf", lw=0,
                        shade=True, rasterized=True)
        ax.set_box_aspect((DRAIN.length + 27, 2 * DRAIN.a, 2 * DRAIN.b), zoom=1.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_xlim(-5, DRAIN.length + 22)
        ax.set_ylim(-DRAIN.a, DRAIN.a)
        ax.set_zlim(0, 2 * DRAIN.b)
        ax.set_title(f"{DESIGN_LABEL[n]}: {pat.n} openings, {pat.open_area_mm2:.0f} mm²", fontsize=8, y=0.86)
    fig.subplots_adjust(wspace=-0.05, hspace=-0.12, left=-0.04, right=1.04, top=1.06, bottom=-0.06)
    save(fig, "fig05_3d")


# ----------------------------------------------------------------------------- closed form
def fig_design_chart():
    opt = [r for r in load("optimize.json") if "error" not in r]
    Hv = 933.3 / (1126 * 9.81) * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(W1, 2.35))
    ax = axes[0]
    al = np.linspace(-10, 40, 300)
    for h_geo, ls, lab in ((0.0, "-", r"$h_{geo}=0$"), (2.5, "--", r"$h_{geo}=2.5$ mm")):
        zl = DRAIN.b * np.cos(np.radians(al)) - DRAIN.port_d / 2 * np.cos(np.radians(al)) + \
            DRAIN.length * np.maximum(np.sin(np.radians(al)), 0)
        need = zl - h_geo
        lstar = np.sqrt(np.clip(Hv / np.maximum(need, 1e-6) - 1, 0, None))
        ax.plot(al, lstar, color=INK, ls=ls, lw=1.4, label=lab)
    ax.set_xlabel(r"tilt $\alpha$ (deg, outlet up)")
    ax.set_ylabel(r"largest open area ratio $\Lambda^*$")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 30)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(r"(a) $\Lambda^*$ for full drainage", loc="left")

    ax = axes[1]
    lam = np.logspace(-1.3, 1.5, 200)
    ax.semilogx(lam, 1 / np.sqrt(1 + 1 / lam ** 2), color=INK, lw=1.4, label="inviscid holes")
    q0 = [(r["Lambda_tot"], r["q0"] / r["q_port"], r["d"]) for r in opt if r["tilt"] == 0]
    q0 = np.array(q0)
    norm = mcolors.Normalize(0.5, 4.0)
    cmap = plt.get_cmap("Blues")
    sc = ax.scatter(q0[:, 0], q0[:, 1], c=q0[:, 2], cmap=mcolors.LinearSegmentedColormap.from_list(
        "b", ["#b5d4f4", "#0c447c"]), norm=norm, s=7, lw=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.02)
    cb.set_label("hole Ø (mm)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel(r"$\Lambda_{tot}$ (all openings)")
    ax.set_ylabel(r"$Q_0 / Q_{max}$")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(b) Flow when submerged", loc="left")

    ax = axes[2]
    lam_tot = np.logspace(-1.3, 1.5, 300)
    for h_geo, ls in ((0.0, "-"), (2.5, "--")):
        # largest tilt with z_lip(alpha) - h_geo <= Hv / (1 + Lambda^2)
        a_star = []
        for lt in lam_tot:
            lift = Hv / (1 + lt ** 2) + h_geo
            f = lambda a: (DRAIN.b - DRAIN.port_d / 2) * math.cos(math.radians(a)) + DRAIN.length * math.sin(math.radians(a)) - lift
            if f(0) > 0:
                a_star.append(np.nan)
                continue
            lo, hi = 0.0, 89.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            a_star.append(lo)
        ax.semilogx(lam_tot, a_star, color=INK, ls=ls, lw=1.4)
    ax.set_xlabel(r"$\Lambda_{tot}$")
    ax.set_ylabel(r"tilt still drained fully (deg)")
    ax.set_ylim(0, 45)
    ax.set_title("(c) Tilt tolerance", loc="left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig06_design_chart")


# ----------------------------------------------------------------------------- results
def fig_curves():
    cur = [r for r in load("curves.json") if "error" not in r]
    show = [("recommended", CAT[0], "-", "Recommended"), ("usz", CAT[1], "--", "USZ as built"),
            ("side2", CAT[2], "-.", "Side rows"), ("cross4", CAT[3], ":", "Four rows"),
            ("ring8", CAT[6], (0, (5, 1.5, 1, 1.5)), "Eight rows")]
    fig, axes = plt.subplots(2, 2, figsize=(W1, 4.3), sharex="col")
    for i, tilt in enumerate((0, 10)):
        for key, col, ls, lab in show:
            r = next(x for x in cur if x.get("design", x["layout"]) == key and x["tilt"] == tilt)
            c = r["curve"]
            h = np.array(c["h"])
            axes[i, 0].plot(h, np.array(c["qL"]) * 6e7, color=col, ls=ls, label=lab)
            axes[i, 1].plot(h, -np.array(c["pg"]) / MMHG, color=col, ls=ls, label=lab)
            axes[i, 0].plot(r["h_sim"], 0, "o", color=col, ms=4, clip_on=False, zorder=5)
        lip = r["lip"]
        for ax in axes[i]:
            ax.axvline(lip, color=MUTED, lw=0.7, ls="--")
            ax.invert_xaxis() if not ax.xaxis_inverted() else None
        axes[i, 0].text(lip, axes[i, 0].get_ylim()[1] * 0.95 if i == 0 else 900, " lip", fontsize=6.8, color=MUTED, va="top")
        axes[i, 0].set_ylabel(f"tilt {tilt}°\nliquid flow (mL/min)")
        axes[i, 1].set_ylabel("vacuum inside, $-p_g$ (mmHg)")
        axes[i, 0].set_ylim(bottom=0)
        axes[i, 1].set_ylim(0, 7.5)
    axes[1, 0].set_xlabel("pool level h above the lowest point (mm)")
    axes[1, 1].set_xlabel("pool level h above the lowest point (mm)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05),
               fontsize=7)
    axes[0, 0].set_title("(a) Liquid drained; dots mark the stop level", loc="left")
    axes[0, 1].set_title("(b) Vacuum inside the drain (at the bore)", loc="left")
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    save(fig, "fig07_drainage_curves")


def _sweep():
    return [r for r in load("sweep.json") if "error" not in r]


def fig_hstop_vs_d(family, name, title):
    sw = _sweep()
    fig, axes = plt.subplots(2, 5, figsize=(W1, 3.4), sharex=True, sharey=True)
    for i, k in enumerate(PATTERN_KEYS):
        ax = axes.flat[i]
        for t in TILTS:
            rr = sorted([r for r in sw if r["layout"] == k and r["family"] == family and r["tilt"] == t],
                        key=lambda r: r["d"])
            ax.plot([r["d"] for r in rr], [r["h_sim"] for r in rr], color=tilt_color(t), lw=1.2,
                    marker="o", ms=2.2)
        ax.set_title(f"{chr(65 + i)} " + LAYOUTS[k][0].replace(", ", ",\n", 1), fontsize=6.8, loc="left")
        ax.set_xticks([1, 2, 3, 4])
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("hole Ø (mm)", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("stop level (mm)", fontsize=7.5)
    handles = [Line2D([], [], color=tilt_color(t), lw=1.4, label=f"{t:+d}°" if t else "0°") for t in TILTS]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.04),
               title="tilt (outlet up > 0)", title_fontsize=7)
    fig.suptitle("")
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    save(fig, name)


def fig_heatmaps():
    sw = _sweep()
    rows = ["usz"] + PATTERN_KEYS
    names = ["USZ as built"] + [f"{chr(65 + i)} {LAYOUTS[k][0]}" for i, k in enumerate(PATTERN_KEYS)]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 3.6), sharey=True)
    cmap = mcolors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#85b7eb", "#185fa5", "#042c53"])
    for j, fam in enumerate(("fixed_count", "fixed_area")):
        M = np.full((len(rows), len(TILTS)), np.nan)
        for i, k in enumerate(rows):
            for jj, t in enumerate(TILTS):
                rr = [r for r in sw if r["layout"] == k and r["tilt"] == t and
                      (k == "usz" or (r["family"] == fam and abs(r["d"] - 1.0) < 1e-9))]
                if rr:
                    M[i, jj] = rr[0]["h_sim"]
        ax = axes[j]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=30, aspect="auto")
        for i in range(M.shape[0]):
            for jj in range(M.shape[1]):
                v = M[i, jj]
                ax.text(jj, i, f"{v:.1f}", ha="center", va="center", fontsize=6.2,
                        color="white" if v > 14 else INK)
        ax.set_xticks(range(len(TILTS)))
        ax.set_xticklabels([f"{t:+d}" if t else "0" for t in TILTS])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names, fontsize=6.8)
        ax.grid(False)
        ax.set_xlabel("tilt (deg, outlet up > 0)")
        ax.set_title("(a) Ø1.0, 19 holes per row" if j == 0 else "(b) Ø1.0, open area 28 mm²", loc="left")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("stop level h (mm above lowest point)", fontsize=7)
    save(fig, "fig10_heatmap_tilt")


def fig_parity():
    sw = _sweep()
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.8))
    for ax, key, lab in ((axes[0], "h_ana", "refined closed form, Eq. (18)"),
                         (axes[1], "h_ana_simple", "simple bound, Eq. (16)")):
        for t in TILTS:
            rr = [r for r in sw if r["tilt"] == t]
            ax.scatter([r[key] for r in rr], [r["h_sim"] for r in rr], s=6, color=tilt_color(t), lw=0,
                       label=f"{t:+d}°" if t else "0°", alpha=0.85)
        mx = max(max(r["h_sim"] for r in sw), max(r[key] for r in sw)) * 1.05
        ax.plot([0, mx], [0, mx], color=INK, lw=0.7)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_aspect("equal")
        ax.set_xlabel(f"stop level, {lab} (mm)")
        ax.set_ylabel("stop level, network simulation (mm)")
        err = np.array([r["h_sim"] - r[key] for r in sw])
        ax.text(0.04, 0.95, f"n = {len(sw)}\nmean |Δ| = {np.mean(np.abs(err)):.2f} mm\n"
                f"max |Δ| = {np.max(np.abs(err)):.2f} mm", transform=ax.transAxes, va="top", fontsize=7)
    axes[1].legend(frameon=False, loc="lower right", fontsize=6.5, title="tilt", title_fontsize=6.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig11_parity")


def fig_optimize():
    opt = [r for r in load("optimize.json") if "error" not in r]
    rng_t = [t for t in TILTS if -10 <= t <= 15]
    rec = recommended_spec()
    rec_area = run_study.build(rec).open_area_mm2
    sizes = sorted(set(r["d"] for r in opt))
    cmap = mcolors.LinearSegmentedColormap.from_list("b", ["#b5d4f4", "#378add", "#0c447c"])
    norm = mcolors.Normalize(min(sizes), max(sizes))
    fig, axes = plt.subplots(1, 3, figsize=(W1, 2.55))
    for d in sizes:
        rr = [r for r in opt if r["d"] == d]
        ns = sorted(set(r["n_x"] for r in rr))
        area, q0, worst = [], [], []
        for n in ns:
            g = [r for r in rr if r["n_x"] == n and r["tilt"] in rng_t]
            area.append(g[0]["area"])
            z = next(r for r in g if r["tilt"] == 0)
            q0.append(z["q0"] / z["q_port"])
            worst.append(max(r["h_ana"] for r in g))
        c = cmap(norm(d))
        lw = 2.0 if d == rec["d"] else 1.0
        axes[0].plot(area, q0, color=c, lw=lw)
        axes[1].plot(area, worst, color=c, lw=lw)
    g = [r for r in opt if r["d"] == rec["d"] and r["n_x"] == rec["n_x"]]
    for ax, val in ((axes[0], None), (axes[1], None)):
        ax.set_xscale("log")
        ax.set_xlabel("total open area (mm²)")
        ax.axvline(2 * DRAIN.port_area * 1e6, color=MUTED, lw=0.6, ls="--")
    if g:
        z = next(r for r in g if r["tilt"] == 0)
        axes[0].plot(rec_area, z["q0"] / z["q_port"], "o", mfc="none", mec=CAT[1], ms=8, mew=1.4)
        axes[1].plot(rec_area, max(r["h_ana"] for r in g if r["tilt"] in rng_t), "o", mfc="none",
                     mec=CAT[1], ms=8, mew=1.4)
    axes[0].set_ylabel(r"$Q_0/Q_{max}$ when submerged")
    axes[0].set_ylim(0, 1.05)
    axes[0].text(2 * DRAIN.port_area * 1e6 * 1.08, 0.05, "2 × bore", fontsize=6.5, color=MUTED)
    axes[0].set_title("(a) Flow", loc="left")
    axes[1].set_ylabel("worst stop level, −10° to +15° (mm)")
    axes[1].set_title("(b) Worst stop level", loc="left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.06, pad=0.02)
    cb.set_label("hole Ø (mm)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    ax = axes[2]
    ang = [r for r in load("angles.json") if "error" not in r]
    for t in (0, 10, 15):
        rr = sorted([r for r in ang if r["tilt"] == t and r["opts"]["floor_gap"] == 0.0], key=lambda r: r["angle"])
        ax.plot([r["angle"] for r in rr], [r["h_sim"] for r in rr], color=tilt_color(t), lw=1.2,
                marker="o", ms=2, label=f"{t:+d}°" if t else "0°")
    ax.set_xlabel(r"row angle $\pm\theta$ from the bottom (deg)")
    ax.set_ylabel("stop level (mm)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", title="tilt", title_fontsize=6.5)
    ax.set_title("(c) Where around the skin", loc="left")
    fig.tight_layout(w_pad=0.7)
    save(fig, "fig12_optimize")


def fig_transient():
    tr = [r for r in load("transient.json") if "error" not in r]
    fig, axes = plt.subplots(3, 2, figsize=(W1, 5.2), sharex="col")
    for j, tilt in enumerate((0, 10)):
        for name in ("recommended", "usz", "side2", "cross4"):
            r = next(x for x in tr if x["design"] == name and x["tilt"] == tilt)
            T = r["trace"]
            t = np.array(T["t"]) / 60
            kw = dict(color=DESIGN_COLOR[name], ls=DESIGN_LS[name], lw=1.4, label=DESIGN_LABEL[name])
            axes[0, j].plot(t, T["h"], **kw)
            axes[1, j].plot(t, -np.array(T["pg"]) / MMHG, **kw)
            axes[2, j].plot(t, np.array(T["qL_out"]) * 6e7, **kw)
            axes[0, j].plot(t[-1], r["h_dyn"], "o", color=DESIGN_COLOR[name], ms=3.5)
        axes[0, j].set_title(f"tilt {tilt}°", loc="left")
        axes[0, j].axhline(r["lip"], color=MUTED, lw=0.7, ls="--")
        axes[2, j].set_xlabel("time (min)")
        axes[1, j].set_ylim(0, 7.5)
        axes[0, j].set_ylim(bottom=0)
        axes[2, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("pool level h (mm)")
    axes[1, 0].set_ylabel("vacuum inside (mmHg)")
    axes[2, 0].set_ylabel("liquid out (mL/min)")
    axes[0, 1].legend(frameon=False, fontsize=6.8, loc="upper right")
    fig.tight_layout(h_pad=0.4, w_pad=1.0)
    save(fig, "fig13_transient")


def fig_sensitivity():
    se = [r for r in load("sensitivity.json") if "error" not in r]
    cases = list(dict.fromkeys(r["case"] for r in se))
    fig, axes = plt.subplots(1, 2, figsize=(W1, 3.3), sharey=True)
    off = {"recommended": -0.24, "usz": -0.08, "side2": 0.08, "cross4": 0.24}
    mk = {"recommended": "o", "usz": "s", "side2": "^", "cross4": "D"}
    for j, tilt in enumerate((0, 10)):
        ax = axes[j]
        for name in off:
            rr = {r["case"]: r for r in se if r["design"] == name and r["tilt"] == tilt}
            ys = [i + off[name] for i, c in enumerate(cases)]
            xs = [rr[c]["h_sim"] if c in rr else np.nan for c in cases]
            ax.scatter(xs, ys, color=DESIGN_COLOR[name], marker=mk[name], s=14, lw=0,
                       label=DESIGN_LABEL[name], zorder=3)
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels(cases, fontsize=7)
        ax.set_xlabel("stop level h (mm)")
        ax.set_title(f"tilt {tilt}°", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_ylim(len(cases) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.55, 1.06),
               fontsize=7)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig14_sensitivity")


def fig_montecarlo():
    mc = [r for r in load("montecarlo.json") if "error" not in r]
    names = ["recommended", "usz", "side2", "cross4"]
    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    rng = np.random.default_rng(1)
    for i, n in enumerate(names):
        v = np.array([r["h_sim"] for r in mc if r["design"] == n])
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, v.size), s=5, color=DESIGN_COLOR[n], lw=0, alpha=0.6)
        q = np.percentile(v, [5, 50, 95])
        ax.plot([q[0], q[2]], [i, i], color=INK, lw=0.8)
        ax.plot(q[1], i, "|", color=INK, ms=10, mew=1.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DESIGN_LABEL[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("stop level h (mm)")
    ax.set_title("(a) 200 random conditions per design", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    for n in names:
        v = np.sort([r["h_sim"] for r in mc if r["design"] == n])
        ax.plot(v, np.arange(1, v.size + 1) / v.size, color=DESIGN_COLOR[n], ls=DESIGN_LS[n],
                label=DESIGN_LABEL[n])
    ax.set_xlabel("stop level h (mm)")
    ax.set_ylabel("fraction of conditions")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax.set_title("(b) Cumulative", loc="left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig15_montecarlo")


def fig_tiltlimit():
    tl = [r for r in load("tiltlimit.json") if "error" not in r]
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    others = [k for k in PATTERN_KEYS]
    for k in others:
        rr = sorted([r for r in tl if r["design"] == k], key=lambda r: r["tilt"])
        ax.plot([r["tilt"] for r in rr], [r["h_sim"] for r in rr], color="#c3c2b7", lw=0.9, zorder=1)
    for k, col, ls, lab in (("usz", CAT[1], "--", "USZ as built (468 mm²)"),
                            ("side2", CAT[2], "-.", "Side rows (28 mm²)"),
                            ("recommended", CAT[0], "-", "Recommended (28 mm²)")):
        rr = sorted([r for r in tl if r["design"] == k], key=lambda r: r["tilt"])
        ax.plot([r["tilt"] for r in rr], [r["h_sim"] for r in rr], color=col, ls=ls, lw=1.8, label=lab, zorder=3)
        ax.plot([r["tilt"] for r in rr], [r["h_geo"] for r in rr], color=col, lw=0.6, ls=":", zorder=2)
    ax.plot([], [], color="#c3c2b7", lw=0.9, label="other layouts A to J (Ø1.0, 25 to 28 mm²)")
    ax.plot([], [], color=MUTED, lw=0.6, ls=":", label="lowest hole edge (geometric limit)")
    ax.set_xlabel(r"tilt $\alpha$ (deg; outlet up $>0$)")
    ax.set_ylabel("stop level h (mm)")
    ax.set_ylim(0, 45)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    save(fig, "fig16_tiltlimit")


def fig_pump():
    pp = [r for r in load("pump.json") if "error" not in r]
    caps = [2.0, 5.0, 10.0, 20.0, float("inf")]
    ramp = ["#b5d4f4", "#85b7eb", "#378add", "#185fa5", "#042c53"]
    fig, axes = plt.subplots(1, 3, figsize=(W1, 2.4), sharey=True)
    for ax, t in zip(axes, (5, 10, 15)):
        for c, col in zip(caps, ramp):
            rr = sorted([r for r in pp if r["q_max_lpm"] == c and r["tilt"] == t], key=lambda r: r["area"])
            lab = "ideal regulator" if math.isinf(c) else f"{c:g} L/min"
            ax.plot([r["area"] for r in rr], [r["h_sim"] for r in rr], color=col, lw=1.4, marker="o", ms=2.2,
                    label=lab)
        ax.set_xscale("log")
        ax.set_xticks([3, 10, 30, 60])
        ax.set_xticklabels(["3", "10", "30", "60"])
        ax.minorticks_off()
        ax.set_xlabel("total open area (mm²)")
        ax.set_title(f"tilt +{t}°", loc="left")
        ax.axvline(2 * DRAIN.port_area * 1e6, color=MUTED, lw=0.6, ls="--")
    axes[0].set_ylabel("stop level h (mm)")
    axes[0].legend(frameon=False, fontsize=6.3, loc="upper left", title="source capacity", title_fontsize=6.3)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig17_pump")


FIGS = {
    "tiltlimit": fig_tiltlimit, "pump": fig_pump,
    "geometry": fig_geometry, "mechanism": fig_mechanism, "patterns": fig_patterns, "sizes": fig_sizes,
    "3d": fig_3d, "design": fig_design_chart, "curves": fig_curves,
    "dcount": lambda: fig_hstop_vs_d("fixed_count", "fig08_hstop_vs_d_count", ""),
    "darea": lambda: fig_hstop_vs_d("fixed_area", "fig09_hstop_vs_d_area", ""),
    "heatmap": fig_heatmaps, "parity": fig_parity, "optimize": fig_optimize,
    "transient": fig_transient, "sensitivity": fig_sensitivity, "montecarlo": fig_montecarlo,
}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(FIGS)):
        FIGS[n]()

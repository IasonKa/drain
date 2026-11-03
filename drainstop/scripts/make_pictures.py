"""Pictures of the drain while the pool empties, from the two-phase network simulation.

    py -3.12 scripts/make_pictures.py [tilt_deg] [--stills]    # stills + animation, default tilt 10

Each picture shows the as-built USZ field and the recommended layout side by side, each in
its own identical pool:
- a cut-away side view in the world frame: pool liquid, drain wall, liquid and air inside the
  drain and the outlet tube, and the near-side openings coloured by what flows through them;
- a 3D view from below: wetted skin and every opening coloured the same way;
- the pool level against time for both designs.

These show the state computed by the reduced-order model (drainstop/model.py): the flow of
liquid and air through every opening, the gas pressure and the liquid level inside the drain.
They are not CFD fields; the 3D CFD check is in vof/.

Output: pictures/*.png and pictures/drain_emptying_tilt<α>.gif
"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches, colors as mcolors  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import run_study  # noqa: E402

OUT = os.path.join(ROOT, "pictures")
WORKERS = max(1, (os.cpu_count() or 2) - 2)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.edgecolor": "#5f5e5a", "axes.labelcolor": "#2c2c2a",
    "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a", "xtick.labelsize": 8, "ytick.labelsize": 8,
})
INK, MUTED = "#2c2c2a", "#898781"
POOL, POOL_EDGE, AIR_IN = "#bcd8f3", "#2a78d6", "#f7f6f2"
WALL, WALL_EDGE, FLOOR = "#9a988f", "#5f5e5a", "#e6dcc8"
SKIN = "#dcdad2"
# opening classes: 0 no flow, 1 liquid in, 2 air in, 3 liquid out
CLS_COLOR = ["#5f5e5a", "#1d5fb4", "#eb6834", "#7b3fb8"]
CLS_LABEL = ["no flow", "liquid in", "air in", "liquid out"]
DESIGN_COLOR = {"usz": "#eb6834", "recommended": "#2a78d6"}
TUBE_LEN, TUBE_WALL = 16.0, 0.8


# ----------------------------------------------------------------------------- simulation
def designs(tilt):
    return {
        "usz": ("USZ as built: 318 × Ø1.0 through both faces (609 openings, 468 mm²)",
                dict(layout="usz", d=1.0, family="reference", tilt=tilt)),
        "recommended": ("Recommended: 2 staggered rows at ±30°, 36 × Ø1.0 (28 mm²)",
                        dict(run_study.best_spec(), tilt=tilt)),
    }


def opening_flows(m, H, st):
    """Liquid in, air in and liquid out (m3/s) through every opening for a solved state."""
    fs, fe, zs, ze = m.exposure(H)
    pg, zeta = st["pg"], st["zeta"]
    if st["mode"] == 1:                                       # interior full of liquid
        def p_in(z):
            return pg + m.rgl * (m.Z_pc - z)
        wet = np.ones(m.pattern.n, bool)
    else:
        def p_in(z):
            return pg + m.rgl * np.maximum(zeta - z, 0.0)
        wet = ze < zeta
    qL = m._liquid_in(H, fs, zs, p_in(zs))
    qG = m._gas_in(fe, st["breached"], p_in(ze))
    qX = m._leak_out(fe, p_in(ze), wet)
    return qL, qG, qX


def classify(qL, qG, qX):
    cls = np.zeros(qL.size, int)
    big = np.stack([np.maximum(qL, 0) / 1e-9, qG / 1e-7, qX / 1e-9])   # 0.06 mL/min, 6 mL/min of air
    on = big.max(0) > 1.0
    cls[on] = 1 + np.argmax(big[:, on], axis=0)
    return cls


def timeline(m):
    """Quasi-static drainage: pool level against time, and the stop."""
    r = m.descend(dh=0.1)
    ok = np.isfinite(r["t"])
    h, t = r["h"][ok], r["t"][ok]
    t_stop = float(np.interp(r["h_stop"] + 0.25, h[::-1], t[::-1]))
    return dict(h=h, t=t, h_stop=r["h_stop"], t_stop=t_stop)


def snapshots(m, tl, times):
    """Solved state at every frame time, marching down so that breached openings stay open."""
    breached = np.zeros(m.pattern.n, bool)
    out = []
    for tt in times:
        if tt >= tl["t_stop"]:
            h, stopped = tl["h_stop"], True
        else:
            h, stopped = float(np.interp(tt, tl["t"], tl["h"])), False
        H = h + m.Z_min
        st = m.state(H, breached)
        breached = st["breached"]
        qL, qG, qX = opening_flows(m, H, st)
        if stopped:
            qL = np.zeros_like(qL)
        zeta = st["zeta"]
        if st["mode"] == 1:
            zeta = math.inf
        # liquid depth in the outlet tube: the liquid share of the bore, as a layer on its floor
        if st["mode"] == 1 or st["qGp"] <= 0:
            tube_depth = m.drain.port_d
        elif stopped or st["qLp"] <= 0:
            tube_depth = 0.0
        else:
            tube_depth = float(np.clip((st["zeta"] - m.Z_lip) / m.ca, 0, m.drain.port_d))
        out.append(dict(t=tt, h=h, H=H, mode=st["mode"], pg=st["pg"], zeta=zeta,
                        qL=0.0 if stopped else st["qL"], qG=st["qG"], stopped=stopped,
                        tube_depth=tube_depth, cls=classify(qL, qG, qX)))
    return out


# ----------------------------------------------------------------------------- geometry
def to_world(m, x, z):
    """Drain frame (x from the tip, z up from the bottom line, mm) to world (X across, Z up)."""
    x, z = np.asarray(x, float), np.asarray(z, float)
    return x * m.ca - z * m.sa, x * m.sa + z * m.ca


def rect(m, x0, x1, z0, z1):
    X, Z = to_world(m, [x0, x1, x1, x0], [z0, z0, z1, z1])
    return np.column_stack([X, Z])


def clip_below(poly, level):
    """Part of a convex polygon (world X, Z) below the horizontal line Z = level."""
    if not math.isfinite(level):
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin, qin = p[1] <= level, q[1] <= level
        if pin:
            out.append(p)
        if pin != qin:
            f = (level - p[1]) / (q[1] - p[1])
            out.append(p + f * (q - p))
    return np.array(out) if len(out) >= 3 else None


def mesh3d(m):
    """Skin facets in world coordinates, their shading and the opening each facet belongs to."""
    dr, ell, pat = m.drain, m.ell, m.pattern
    P = ell.perimeter
    nx, ns = 480, 200
    xe = np.linspace(0, dr.length, nx)
    se = np.linspace(-P / 2, P / 2, ns)
    S, Xl = np.meshgrid(se, xe)
    Y, Zl = ell.yz(ell.theta(S))
    X, Z = to_world(m, Xl, Zl)
    xc, sc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (se[1:] + se[:-1])
    thc = ell.theta(sc)
    zc_loc = dr.b * (1 - np.cos(thc))
    Zc = xc[:, None] * m.sa + zc_loc[None, :] * m.ca               # world height of facet centres
    nl = np.stack([np.sin(thc) / dr.a, -np.cos(thc) / dr.b], -1)    # local (y, z) outward normal
    nl /= np.linalg.norm(nl, axis=-1, keepdims=True)
    nrm = np.stack([-nl[:, 1] * m.sa, nl[:, 0], nl[:, 1] * m.ca], -1)
    light = np.array([0.35, -0.6, -0.72])
    light /= np.linalg.norm(light)
    shade = 0.58 + 0.42 * np.clip(nrm @ light, 0, 1)
    owner = -np.ones((nx - 1, ns - 1), int)
    s_h = ell.s_of(pat.theta)
    for i, (x0, s0, d, a) in enumerate(zip(pat.x, s_h, pat.d, pat.area_scale)):
        r = 0.5 * d * math.sqrt(a)
        ix = np.where(np.abs(xc - x0) <= r)[0]
        for ss in (s0, s0 - P, s0 + P):
            js = np.where(np.abs(sc - ss) <= r)[0]
            if ix.size and js.size:
                inside = (xc[ix, None] - x0) ** 2 + (sc[None, js] - ss) ** 2 <= r * r
                blk = owner[np.ix_(ix, js)]
                blk[inside] = i
                owner[np.ix_(ix, js)] = blk
    # outlet tube, along the drain axis at the bore centre
    ang = np.linspace(0, 2 * np.pi, 48)
    xt, at = np.meshgrid(np.linspace(dr.length, dr.length + TUBE_LEN, 4), ang)
    rt = dr.port_d / 2 + TUBE_WALL
    Xt, Zt = to_world(m, xt, dr.b + rt * np.sin(at))
    return dict(X=X, Y=Y, Z=Z, Zc=Zc, shade=shade, owner=owner,
                tube=(Xt, rt * np.cos(at), Zt))


# ----------------------------------------------------------------------------- drawing
def draw_side(ax, m, snap, title):
    dr = m.drain
    L, b, w, rp = dr.length, dr.b, dr.wall, dr.port_d / 2
    Xend, _ = to_world(m, L + TUBE_LEN, b)
    x_lo, x_hi = -6.0, float(Xend) + 4.0
    z_lo = m.Z_min - 3.0
    z_hi = m.Z_top + 13.5                                               # start level + room for text
    ax.add_patch(patches.Rectangle((x_lo, z_lo), x_hi - x_lo, 0.6, color=FLOOR, lw=0, zorder=0))
    ax.add_patch(patches.Rectangle((x_lo, z_lo + 0.6), x_hi - x_lo, snap["H"] - z_lo - 0.6,
                                   color=POOL, lw=0, zorder=1))
    ax.plot([x_lo, x_hi], [snap["H"]] * 2, color=POOL_EDGE, lw=1.0, zorder=2)

    def fill(poly, color, z, **kw):
        if poly is not None:
            ax.add_patch(patches.Polygon(poly, closed=True, color=color, zorder=z, **kw))

    zeta = snap["zeta"]
    fill(rect(m, 0, L, 0, 2 * b), WALL, 3, lw=0)                       # skin, tip cap, end face
    channels = [rect(m, w, L - w, w, 2 * b - w),                        # lumen
                rect(m, L - w - 0.05, L + 0.05, b - rp, b + rp)]        # bore through the end face
    tube_out = rect(m, L, L + TUBE_LEN, b - rp - TUBE_WALL, b + rp + TUBE_WALL)
    fill(tube_out, WALL, 3, lw=0)
    for c in channels:
        fill(c, AIR_IN, 4, lw=0)
        fill(clip_below(c, zeta), POOL, 5, lw=0)
    fill(rect(m, L, L + TUBE_LEN, b - rp, b + rp), AIR_IN, 4, lw=0)                # tube interior
    if snap["tube_depth"] > 0:
        fill(rect(m, L, L + TUBE_LEN, b - rp, b - rp + snap["tube_depth"]), POOL, 5, lw=0)
    for poly in (rect(m, 0, L, 0, 2 * b), tube_out):
        ax.add_patch(patches.Polygon(poly, closed=True, fill=False, ec=WALL_EDGE, lw=0.6, zorder=6))
    if math.isfinite(zeta) and snap["mode"] != 1:                      # inner free surface
        lum = clip_below(channels[0], zeta)
        if lum is not None:
            top = lum[np.abs(lum[:, 1] - zeta) < 1e-6]
            if len(top) >= 2:
                ax.plot(top[:, 0], top[:, 1], color=POOL_EDGE, lw=0.9, zorder=6)

    # near-side openings (seen from -y), drawn on the skin, coloured by what flows
    pat = m.pattern
    near = np.sin(pat.theta) <= 1e-9
    th = pat.theta[near]
    n_y, n_z = np.sin(th) / dr.a, -np.cos(th) / dr.b                  # outward normal (unnormalised)
    fy = np.abs(n_y) / np.hypot(n_y, n_z)                               # foreshortening seen from -y
    zloc = b * (1 - np.cos(th))
    Xh, Zh = to_world(m, pat.x[near], zloc)
    for Xi, Zi, di, fi, ci in zip(Xh, Zh, pat.d[near], fy, snap["cls"][near]):
        ax.add_patch(patches.Ellipse((Xi, Zi), di, di * max(fi, 0.35), angle=m.tilt,
                                     color=CLS_COLOR[ci], lw=0, zorder=7))

    # outflow at the tube end
    Xo, Zo = to_world(m, L + TUBE_LEN + 0.5, b)
    ax.annotate("", xy=(Xo + 7 * m.ca, Zo + 7 * m.sa), xytext=(Xo, Zo),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0), zorder=8)
    ax.text(Xo + 2, Zo - 5.2, "to suction\n−7 mmHg", fontsize=7, color=MUTED, ha="left", va="top")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(z_lo, z_hi)
    ax.set_aspect("equal")
    ax.axis("off")
    if snap["stopped"]:
        status = ("STOPPED: pool drained down to the lowest openings (the vacuum holds the liquid "
                  "left inside up to the bore lip)") if snap["h"] < 3 else \
                 "STOPPED: air through the exposed openings leaves too little vacuum to lift liquid to the outlet"
        scol = "#1b7a4e" if snap["h"] < 3 else "#c43b3a"
    elif snap["mode"] == 1:
        status, scol = "draining, drain full of liquid", INK
    elif snap["qG"] * 6e4 < 0.05:
        status, scol = "draining, air pocket at atmospheric pressure in the top of the drain", INK
    else:
        status, scol = "draining, air entering through the exposed openings and leaving with the liquid", INK
    ax.text(x_lo + 1, z_hi - 0.5, title, fontsize=8.5, weight="bold", color=INK, va="top")
    ax.text(x_lo + 1, z_hi - 4.3,
            f"pool level {snap['h']:5.1f} mm    liquid out {snap['qL'] * 6e7:4.0f} mL/min    "
            f"air in {snap['qG'] * 6e4:4.1f} L/min    pressure inside {snap['pg']:+5.0f} Pa",
            fontsize=7.8, color=INK, va="top", family="DejaVu Sans Mono")
    ax.text(x_lo + 1, z_hi - 8.1, status, fontsize=7.8, color=scol, va="top")


def draw_3d(ax, m, g, snap):
    base = np.array(mcolors.to_rgb(SKIN))
    wet = np.array(mcolors.to_rgb("#8fbde8"))
    wetted = g["Zc"] < snap["H"]
    fc = np.where(wetted[..., None], wet, base) * g["shade"][None, :, None]
    own = g["owner"]
    has = own >= 0
    cls_rgb = np.array([mcolors.to_rgb(c) for c in CLS_COLOR])
    fc[has] = cls_rgb[snap["cls"][own[has]]]
    fc = np.concatenate([fc, np.ones(fc.shape[:2] + (1,))], -1)
    skin = ax.plot_surface(g["X"], g["Y"], g["Z"], facecolors=fc, shade=False, rstride=1, cstride=1,
                           linewidth=0, antialiased=False, rasterized=True)
    tube = ax.plot_surface(*g["tube"], color="#bdbbb3", lw=0, shade=True, rasterized=True)
    for s in (skin, tube):                     # the 3D box is narrower than the drawing at this zoom
        s.set_clip_on(False)
    X, Z = g["X"], g["Z"]
    xr = (X.min() - 2, X.max() + TUBE_LEN + 2)
    zr = (Z.min(), max(Z.max(), g["tube"][2].max()))
    ax.set_xlim(*xr)
    ax.set_ylim(-m.drain.a, m.drain.a)
    ax.set_zlim(*zr)
    ax.set_box_aspect((xr[1] - xr[0], 2 * m.drain.a, zr[1] - zr[0]), zoom=1.45)
    ax.view_init(elev=-26, azim=-82)
    ax.set_axis_off()


def draw_levels(ax, ctx, t_now):
    for key in ("usz", "recommended"):
        tl = ctx["tl"][key]
        t = np.append(tl["t"], [tl["t_stop"], ctx["t_end"]])
        h = np.append(tl["h"], [tl["h_stop"], tl["h_stop"]])
        o = np.argsort(t)
        lab = "USZ as built" if key == "usz" else "Recommended"
        ax.plot(t[o], h[o], color=DESIGN_COLOR[key], lw=1.6, label=lab)
        h_now = tl["h_stop"] if t_now >= tl["t_stop"] else float(np.interp(t_now, tl["t"], tl["h"]))
        ax.plot([t_now], [h_now], "o", color=DESIGN_COLOR[key], ms=5, mec="white", mew=0.8, zorder=5)
    ax.axvline(t_now, color=MUTED, lw=0.8)
    ax.set_xlim(0, ctx["t_end"])
    ax.set_ylim(0, None)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("pool level (mm)", fontsize=8)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False)


def legend_strip(ax):
    ax.axis("off")
    items = [(POOL, "liquid (pool and inside the drain)", "s"), (AIR_IN, "air inside the drain", "s"),
             (WALL, "drain wall", "s")] + [(CLS_COLOR[i], "opening: " + CLS_LABEL[i], "o") for i in (1, 2, 0)]
    for k, (c, lab, mk) in enumerate(items):
        x = 0.01 + k * 0.165
        ax.plot([x], [0.5], mk, color=c, ms=8, mec=WALL_EDGE if c == AIR_IN else c, mew=0.6,
                transform=ax.transAxes)
        ax.text(x + 0.012, 0.5, lab, fontsize=7.5, va="center", color=INK, transform=ax.transAxes)


def render(ctx, i, path=None, dpi=100):
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.42, 0.1], width_ratios=[0.64, 0.36],
                          left=0.01, right=0.99, top=0.93, bottom=0.03, hspace=0.12, wspace=0.0)
    t_now = ctx["times"][i]
    fig.suptitle(f"Drain emptying a glycerol–water pool at −7 mmHg, outlet end {ctx['tilt']:g}° up"
                 f"   ·   t = {t_now:5.1f} s", fontsize=11, color=INK, y=0.975)
    fig.text(0.5, 0.943, "Two-phase network simulation (reduced-order model, not CFD); "
             "each drain in its own identical 200 × 80 mm pool; side view cut through the axis, "
             "openings on the near side", fontsize=7.5, color=MUTED, ha="center")
    for r, key in enumerate(("usz", "recommended")):
        m, snap = ctx["models"][key], ctx["snaps"][key][i]
        draw_side(fig.add_subplot(gs[r, 0]), m, snap, ctx["titles"][key])
        draw_3d(fig.add_subplot(gs[r, 1], projection="3d"), m, ctx["mesh"][key], snap)
    ax = fig.add_subplot(gs[2, :])
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.05, pos.y0 + 0.02, pos.width - 0.08, pos.height - 0.02])
    draw_levels(ax, ctx, t_now)
    legend_strip(fig.add_subplot(gs[3, :]))
    if path:
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return None
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# ----------------------------------------------------------------------------- driver
_CTX = None


def _init(ctx):
    global _CTX
    _CTX = ctx


def _frame(i):
    return render(_CTX, i, dpi=90)


def _still(args):
    i, path = args
    render(_CTX, i, path=path, dpi=150)
    return path


def build_context(tilt, n_frames=84, n_hold=14):
    ctx = dict(tilt=tilt, models={}, titles={}, tl={}, mesh={}, snaps={})
    for key, (title, spec) in designs(tilt).items():
        m = run_study.make_model(spec)
        ctx["models"][key], ctx["titles"][key] = m, title
        ctx["tl"][key] = timeline(m)
        ctx["mesh"][key] = mesh3d(m)
        print(f"  {key}: stops at {ctx['tl'][key]['h_stop']:.2f} mm after {ctx['tl'][key]['t_stop']:.1f} s")
    t_end = 1.08 * max(tl["t_stop"] for tl in ctx["tl"].values())
    ctx["t_end"] = t_end
    # a frame at every stop, then evenly spaced in time
    times = np.linspace(0, t_end, n_frames)
    times = np.sort(np.concatenate([times, [tl["t_stop"] for tl in ctx["tl"].values()]]))
    times = np.concatenate([times, np.full(n_hold, times[-1])])
    ctx["times"] = times
    for key, m in ctx["models"].items():
        ctx["snaps"][key] = snapshots(m, ctx["tl"][key], times)
    return ctx


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tilt = float(args[0]) if args else 10.0
    os.makedirs(OUT, exist_ok=True)
    print("[simulate]")
    ctx = build_context(tilt)
    times = ctx["times"]
    t_usz, t_rec = ctx["tl"]["usz"]["t_stop"], ctx["tl"]["recommended"]["t_stop"]
    tag = f"tilt{tilt:g}"

    def nearest(t):
        return int(np.argmin(np.abs(times[: len(times) - 14] - t)))
    # first frame where the USZ drain draws air (more than 1 L/min)
    first_air = next(i for i, s in enumerate(ctx["snaps"]["usz"]) if s["qG"] * 6e4 > 1.0)
    stills = [(0, f"{tag}_1_submerged.png"), (first_air, f"{tag}_2_air_enters.png"),
              (nearest(t_usz), f"{tag}_3_usz_stops.png"), (nearest(t_rec), f"{tag}_4_end.png")]
    print("[stills]")
    with Pool(WORKERS, initializer=_init, initargs=(ctx,)) as pool:
        for p in pool.map(_still, [(i, os.path.join(OUT, name)) for i, name in stills]):
            print("  ", os.path.basename(p))
        if "--stills" in sys.argv:
            return
        print(f"[animation] {len(times)} frames")
        frames = pool.map(_frame, range(len(times)), chunksize=2)
    h, w = frames[0].shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    im = ax.imshow(frames[0])
    gif = os.path.join(OUT, f"drain_emptying_{tag}.gif")
    writer = PillowWriter(fps=8)
    with writer.saving(fig, gif, dpi=100):
        for f in frames:
            im.set_data(f)
            writer.grab_frame()
    plt.close(fig)
    print("  ", os.path.basename(gif), f"{os.path.getsize(gif) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

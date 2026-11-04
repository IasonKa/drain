"""Pictures and numbers of a Kratos two-fluid drain run, next to the model.

    python vof/kratos/render_kratos.py <run dir> [--avg 0.03] [--frames]

Reads <run>/vtk_output/*.vtk (VELOCITY, PRESSURE, DISTANCE at the nodes; DISTANCE < 0 is liquid),
<run>/monitor.csv and <run>/model_state.json. Writes into <run>/pictures/:
  midplane.png      liquid/air, pressure, velocity in the plane through the drain axis (world frame)
  outlet_end.png    the outlet end, with streamlines
  row.png           the plane through a row of openings
  sections.png      cross-sections through a submerged and an exposed opening
  openings.png      flow through every opening on the unrolled skin, CFD next to the model
  view3d.png        3D: pressure on the drain, the liquid surface, the flow through every opening
  history.png       outlet liquid and air flow and the pressure inside the drain against time
  summary.json      averaged CFD numbers next to the model's
With --frames also frames/ of the mid-plane over time and an animated GIF.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOF = os.path.dirname(HERE)
ROOT = os.path.dirname(VOF)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import run_study  # noqa: E402
from make_case import SPECS, TILT, TUBE_LEN, hole_frames  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.edgecolor": "#5f5e5a", "axes.labelcolor": "#2c2c2a", "axes.titlesize": 9.5,
    "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
INK, MUTED, SOLID = "#2c2c2a", "#898781", "#7b7a74"
PHASE = mcolors.LinearSegmentedColormap.from_list("phase", ["#fbfaf6", "#9ec8ef", "#2a78d6"])
VEL_NORM = mcolors.LogNorm(1e-3, 40)
EPS = 0.15e-3                       # m, half-width of the drawn interface


# ----------------------------------------------------------------------------- data
def vtk_time(path):
    """Kratos names the files <model part>_<rank>_<time>.vtk with output_control_type 'time'."""
    return float(re.findall(r"_([0-9.eE+-]+)\.vtk$", os.path.basename(path))[0])


def vtk_files(run):
    """All field outputs: vtk_saved/ (before a resume) and vtk_output/ (since), one per time."""
    fs = glob.glob(os.path.join(run, "vtk_saved", "*.vtk")) + glob.glob(os.path.join(run, "vtk_output", "*.vtk"))
    by_time = {vtk_time(f): f for f in fs}
    return [by_time[t] for t in sorted(by_time)]


def read_monitor(run):
    with open(os.path.join(run, "monitor.csv")) as f:
        rows = list(csv.reader(f))
    head, data = rows[0], np.array([[float(v) for v in r] for r in rows[1:]])
    return {k: data[:, i] for i, k in enumerate(head)}


def load(path):
    import pyvista as pv
    g = pv.read(path)
    for k in ("VELOCITY", "PRESSURE", "DISTANCE"):
        if k not in g.point_data:
            raise KeyError(f"{path}: no {k}")
    return g


def file_time(g, path, mon):
    return vtk_time(path)


def average(files):
    """Point-wise average of the last outputs (same mesh)."""
    g = load(files[0]).copy()
    acc = {k: np.array(g.point_data[k], float) for k in ("VELOCITY", "PRESSURE", "DISTANCE")}
    for f in files[1:]:
        h = load(f)
        for k in acc:
            acc[k] += np.asarray(h.point_data[k], float)
    for k in acc:
        g.point_data[k] = acc[k] / len(files)
    return g


class Frame:
    def __init__(self, tilt):
        al = math.radians(tilt)
        self.sa, self.ca = math.sin(al), math.cos(al)

    def world(self, x, z):
        return x * self.ca - z * self.sa, x * self.sa + z * self.ca

    def world_vec(self, ux, uz):
        return ux * self.ca - uz * self.sa, ux * self.sa + uz * self.ca


# ----------------------------------------------------------------------------- cuts
def cut(g, normal, origin):
    s = g.slice(normal=normal, origin=origin).triangulate()
    tri = s.faces.reshape(-1, 4)[:, 1:] if s.n_cells else np.zeros((0, 3), int)
    return s, tri


def phase(d):
    return np.clip(0.5 - 0.5 * d / EPS, 0.0, 1.0)


def heat_panels(fig, axes, X, Y, tri, pd, frame_vec, h_line, xlim, ylim, stream, p_range):
    U = pd["VELOCITY"]
    spd = np.linalg.norm(U, axis=1)
    alpha = phase(pd["DISTANCE"])
    p = pd["PRESSURE"]
    T = mtri.Triangulation(X, Y, tri)
    specs = [("liquid fraction (blue liquid, white air)", alpha, PHASE, mcolors.Normalize(0, 1)),
             ("pressure p (Pa, gauge)", p, "RdBu_r", mcolors.TwoSlopeNorm(0, *p_range)),
             ("velocity magnitude |U| (m/s, log scale)", np.clip(spd, 1e-3, 40), "magma", VEL_NORM)]
    for ax, (lab, val, cmap, norm) in zip(axes, specs):
        pc = ax.tripcolor(T, val, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.01, aspect=18).set_label(lab, fontsize=8)
        if h_line is not None:
            ax.axhline(h_line, color="#2a78d6", lw=0.7, ls=(0, (4, 3)))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_facecolor(SOLID)
    if stream is not None:                       # (gx, gy, u, v) sampled on a regular grid
        gx, gy, iu, iv = stream
        axes[2].streamplot(gx, gy, iu, iv, density=2.2, color="#d6f5ff", linewidth=0.45, arrowsize=0.55)


def sample_grid(g, xlim, ylim, to_drain, n=300):
    """Velocity sampled from the volume on a regular grid of plot coordinates; to_drain maps the
    plot coordinates (mm) to drain-frame points (mm). NaN outside the fluid."""
    import pyvista as pv
    gx = np.linspace(*xlim, n)
    gy = np.linspace(*ylim, max(int(n * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])), 20))
    GX, GY = np.meshgrid(gx, gy)
    P = to_drain(GX.ravel(), GY.ravel()) * 1e-3
    smp = pv.PolyData(P).sample(g)
    U = np.asarray(smp.point_data["VELOCITY"])
    ok = np.asarray(smp.point_data["vtkValidPointMask"], bool)
    U[~ok] = np.nan
    return gx, gy, U.reshape(GX.shape + (3,))


def plane_world(g, frame, st, normal, origin, title, path, crop, stream):
    s, tri = cut(g, normal, origin)
    if not len(tri):
        return
    P = s.points * 1e3
    X, Z = frame.world(P[:, 0], P[:, 2])
    pd = {k: np.asarray(s.point_data[k]) for k in ("VELOCITY", "PRESSURE", "DISTANCE")}
    xlim, zlim = crop
    asp = (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])
    fig, axes = plt.subplots(3, 1, figsize=(12, 3 * 12 * asp * 0.86 + 1.4), constrained_layout=True)
    pr = pd["PRESSURE"]
    lo = min(np.percentile(pr, 0.5), -50.0)
    hi = max(np.percentile(pr, 99.5), 30.0)
    sg = None
    if stream:                                   # plane through the axis (y = 0) only
        def to_drain(Xw, Zw):
            return np.column_stack([Xw * frame.ca + Zw * frame.sa, np.zeros_like(Xw), -Xw * frame.sa + Zw * frame.ca])
        gx, gz, U = sample_grid(g, xlim, zlim, to_drain)
        uX, uZ = frame.world_vec(U[..., 0], U[..., 2])
        sg = (gx, gz, uX, uZ)
    heat_panels(fig, axes, X, Z, tri, pd, None, st["h"], xlim, zlim, sg, (lo, hi))
    for ax in axes:
        ax.set_ylabel("height (mm)")
    axes[-1].set_xlabel("horizontal position (mm)")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def sections(g, st, xs, title, path):
    fig, axes = plt.subplots(len(xs), 3, figsize=(12, 3.6 * len(xs)), constrained_layout=True)
    for r, (x0, lab) in enumerate(xs):
        s, tri = cut(g, (1, 0, 0), (x0 * 1e-3, 0, 0))
        P = s.points * 1e3
        pd = {k: np.asarray(s.point_data[k]) for k in ("VELOCITY", "PRESSURE", "DISTANCE")}
        pr = pd["PRESSURE"]
        lo, hi = min(np.percentile(pr, 0.5), -50.0), max(np.percentile(pr, 99.5), 30.0)
        gy, gz, U = sample_grid(g, (-11, 11), (-2.5, 12),
                                lambda Y, Z: np.column_stack([np.full_like(Y, x0), Y, Z]), n=220)
        heat_panels(fig, axes[r], P[:, 1], P[:, 2], tri, pd, None, None, (-11, 11), (-2.5, 12),
                    (gy, gz, U[..., 1], U[..., 2]), (lo, hi))
        for ax in axes[r]:
            ax.set_title(f"{lab}, x = {x0:.1f} mm from the tip", fontsize=8.5)
            ax.set_xlabel("y (mm)")
            ax.set_ylabel("z, drain frame (mm)")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- openings
def opening_flows(g, m, n_pts=31):
    """Flow (m3/s, into the drain) through every opening: the velocity sampled on equal-area
    points of the opening's mid-wall disc, liquid where DISTANCE < 0."""
    import pyvista as pv
    c, n, r = hole_frames(m)
    mid = c - 0.5 * m.drain.wall * n
    k = np.arange(n_pts) + 0.5
    rho = np.sqrt(k / n_pts)
    phi = k * math.pi * (3 - math.sqrt(5))
    pts, owner = [], []
    for i in range(len(mid)):
        nn = n[i]
        e1 = np.cross(nn, [1.0, 0, 0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(nn, [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nn, e1)
        d = r[i] * 0.95 * (rho[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        pts.append(mid[i] + d)
        owner.append(np.full(n_pts, i))
    pts = np.concatenate(pts) * 1e-3
    owner = np.concatenate(owner)
    smp = pv.PolyData(pts).sample(g)
    U = np.asarray(smp.point_data["VELOCITY"])
    D = np.asarray(smp.point_data["DISTANCE"])
    valid = np.asarray(smp.point_data.get("vtkValidPointMask", np.ones(len(pts))), bool)
    un = -np.einsum("ij,ij->i", U, n[owner])                      # into the drain
    area = math.pi * (r[owner] * 1e-3) ** 2 / n_pts
    q = un * area * valid
    liq = D < 0
    qL = np.bincount(owner, q * liq, minlength=len(mid))
    qG = np.bincount(owner, q * ~liq, minlength=len(mid))
    return qL, qG


def opening_flows_model(m, h):
    from make_pictures import opening_flows as of
    br = np.zeros(m.pattern.n, bool)
    for hh in np.arange(m.h_start(), h, -0.25):
        br = m.state(hh + m.Z_min, br)["breached"]
    H = h + m.Z_min
    st = m.state(H, br)
    qL, qG, qX = of(m, H, st)
    return qL - qX, qG


def openings_figure(m, st, qLc, qGc, qLm, qGm, path, title):
    pat = m.pattern
    s = m.ell.s_of(pat.theta)
    P = m.ell.perimeter
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.2), constrained_layout=True)
    ml = max(np.max(np.abs(qLc)), np.max(np.abs(qLm)), 1e-12) * 6e7
    mg = max(np.max(qGc), np.max(qGm), 1e-12) * 6e4
    th = np.linspace(-np.pi, np.pi, 721)
    zl = m.drain.b * (1 - np.cos(th))
    xw = (st["h"] + m.Z_min - zl * m.ca) / m.sa
    ok = (xw > 0) & (xw < m.drain.length)
    for c, (lab, qL, qG) in enumerate((("CFD (Kratos two-fluid)", qLc, qGc), ("model (network)", qLm, qGm))):
        for r_, (kind, q, vmax, unit, cmap, fmt) in enumerate((
                ("liquid in", qL * 6e7, ml, "mL/min", "Blues", "{:.0f}"),
                ("air in", qG * 6e4, mg, "L/min", "Oranges", "{:.1f}"))):
            ax = axes[r_, c]
            sc = ax.scatter(pat.x, s, c=np.clip(q, 0, None), cmap=cmap, vmin=0, vmax=vmax,
                            s=30 * pat.d ** 2, edgecolors="#5f5e5a", linewidths=0.4)
            fig.colorbar(sc, ax=ax, pad=0.01).set_label(f"{kind} per opening ({unit})", fontsize=8)
            ax.plot(xw[ok], m.ell.s_of(th[ok]), ".", ms=0.8, color="#2a78d6")
            ax.set_xlim(-2, m.drain.length + 2)
            ax.set_ylim(-P / 2, P / 2)
            ax.set_yticks([-P / 2, -P / 4, 0, P / 4, P / 2])
            ax.set_yticklabels(["top", "−y side", "bottom", "+y side", "top"])
            ax.set_xlabel("x from the tip (mm)")
            ax.grid(color="#e1e0d9", lw=0.5)
            ax.set_title(f"{lab}: {kind}, total " + fmt.format(np.sum(np.clip(q, 0, None))) + f" {unit}",
                         fontsize=9)
    fig.suptitle(title + "\nunrolled skin: each dot is one opening; the blue dotted line is the waterline",
                 fontsize=10, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- 3D
def view3d(g, m, st, qLc, qGc, path, title):
    import pyvista as pv
    pv.OFF_SCREEN = True
    dr = m.drain
    a, b, L = dr.a, dr.b, dr.length
    gm = g.copy()
    gm.points = gm.points * 1e3
    surf = gm.extract_surface()
    cc = surf.cell_centers().points
    keep = (np.abs(cc[:, 1]) < a + 1.0) & (cc[:, 2] > -1.0) & (cc[:, 2] < 2 * b + 1.5) & (cc[:, 0] > -1.0)
    drain = surf.extract_cells(np.where(keep)[0]).extract_surface()
    liquid = gm.contour([0.0], scalars="DISTANCE")
    for mesh in (drain, liquid):
        mesh.rotate_y(-TILT, inplace=True)
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1150))
    pl.set_background("white")
    pl.add_mesh(drain, scalars="PRESSURE", cmap="RdBu_r", clim=(-950, 950),
                scalar_bar_args=dict(title="pressure on the drain (Pa)", vertical=True, position_x=0.9,
                                     position_y=0.55, height=0.38, width=0.035, title_font_size=18,
                                     label_font_size=15, color="black", fmt="%.0f"))
    if liquid.n_points:
        pl.add_mesh(liquid, color="#4f97e0", opacity=0.30, smooth_shading=True)
    c, n, r = hole_frames(m)
    P = c + 0.35 * n
    al = math.radians(TILT)
    Pw = np.column_stack([P[:, 0] * math.cos(al) - P[:, 2] * math.sin(al), P[:, 1],
                          P[:, 0] * math.sin(al) + P[:, 2] * math.cos(al)])
    for mask, vals, cmap, lab, fmt, px in ((qLc * 6e7 > 0.05, qLc * 6e7, "Blues", "liquid in per opening (mL/min)", "%.0f", 0.9),
                                           (qGc * 6e4 > 0.01, qGc * 6e4, "Oranges", "air in per opening (L/min)", "%.2f", 0.74)):
        if mask.any():
            cloud = pv.PolyData(Pw[mask])
            cloud["q"] = vals[mask]
            pl.add_mesh(cloud, scalars="q", cmap=cmap, clim=(0, max(vals[mask].max(), 1e-6)), point_size=17,
                        render_points_as_spheres=True,
                        scalar_bar_args=dict(title=lab, vertical=True, position_x=px, position_y=0.08, height=0.38,
                                             width=0.035, title_font_size=18, label_font_size=15, color="black",
                                             fmt=fmt))
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.camera_position = [(55, -125, -125), (62, 0, 12), (0, 0, 1)]
    pl.camera.zoom(1.12)
    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(path)
    pl.close()


# ----------------------------------------------------------------------------- history
def history(mon, st, path, title, avg_from):
    t = mon["t"]
    ql = mon["Q_liquid"] * 6e7
    qg = (mon["Q_total"] - mon["Q_liquid"]) * 6e4
    probes = [k for k in mon if re.fullmatch(r"p\d+", k)]
    P = np.column_stack([mon[k] for k in probes])
    D = np.column_stack([mon["d" + k[1:]] for k in probes])
    gas = np.where(D[:, :5] > 0, P[:, :5], np.nan)                # drain-top probes that are in air
    with np.errstate(all="ignore"):
        pg = np.nanmean(gas, axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    axes[0].plot(t, ql, color="#2a78d6", lw=1)
    axes[0].axhline(st["qL_ml_min"], color="#2a78d6", ls="--", lw=1, label="model")
    axes[1].plot(t, qg, color="#eb6834", lw=1)
    axes[1].axhline(st["qG_l_min"], color="#eb6834", ls="--", lw=1, label="model")
    axes[2].plot(t, pg, color="#6250d6", lw=1, label="CFD, top of the drain interior (in air)")
    axes[2].axhline(st["pg"], color="#6250d6", ls="--", lw=1, label="model gas pressure")
    axes[0].set_ylabel("liquid out\n(mL/min)")
    axes[1].set_ylabel("air out\n(L/min)")
    axes[2].set_ylabel("pressure inside\nthe drain (Pa)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(color="#e1e0d9", lw=0.5)
        ax.axvspan(avg_from, t[-1], color="#f1efe8", zorder=0)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle(title + "\n(shaded: averaging window)", fontsize=10, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    w = t >= avg_from
    with np.errstate(all="ignore"):
        return dict(qL_ml_min=float(np.mean(ql[w])), qG_l_min=float(np.mean(qg[w])),
                    pg=float(np.nanmean(pg[w])) if np.isfinite(pg[w]).any() else None)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--avg", type=float, default=0.03, help="s at the end used for averages")
    ap.add_argument("--frames", action="store_true", help="also an animation of the mid-plane")
    args = ap.parse_args()
    run = os.path.abspath(args.run)
    st = json.load(open(os.path.join(run, "model_state.json")))
    case = st["case"]
    m = run_study.make_model(SPECS[case] or dict(run_study.best_spec(), tilt=TILT))
    frame = Frame(TILT)
    mon = read_monitor(run)
    files = vtk_files(run)
    if not files:
        sys.exit("no field output yet")
    times = [file_time(load(f), f, mon) for f in files[-12:]]
    t_end = times[-1]
    avg_from = max(t_end - args.avg, 0.0)
    sel = [f for f, t in zip(files[-12:], times) if t >= avg_from - 1e-9] or files[-1:]
    g = average(sel)
    out = os.path.join(run, "pictures")
    os.makedirs(out, exist_ok=True)
    label = "USZ as built" if case.startswith("usz") else "Recommended"
    head = (f"{label}, outlet end {TILT:g}° up, pool level {st['h']:g} mm, suction −7 mmHg — Kratos two-fluid "
            f"CFD, average of t = {avg_from:.3g}–{t_end:.3g} s")

    hist = history(mon, st, os.path.join(out, "history.png"), head, avg_from)
    plane_world(g, frame, st, (0, 1, 0), (0, 0, 0), head + "\nplane through the drain axis",
                os.path.join(out, "midplane.png"), ((-8, 140), (-3.5, 36)), False)
    plane_world(g, frame, st, (0, 1, 0), (0, 0, 0), head + "\noutlet end, plane through the axis",
                os.path.join(out, "outlet_end.png"), ((92, 136), (12, 34)), True)
    dr = m.drain
    if case.startswith("rec"):
        th = math.radians(-30.0)
        yr, zr = dr.a * math.sin(th), dr.b * (1 - math.cos(th))
        normal, origin = (0.0, -(zr - dr.b), yr), (0.0, 0.0, dr.b * 1e-3)
    else:
        th = m.pattern.theta[np.argmin(np.abs(m.pattern.theta + 0.089))]
        yr = dr.a * math.sin(th)
        normal, origin = (0.0, 1.0, 0.0), (0.0, yr * 1e-3, 0.0)
    plane_world(g, frame, st, normal, origin, head + "\nplane through a row of openings",
                os.path.join(out, "row.png"), ((-8, 140), (-3.5, 36)), False)

    qLc, qGc = opening_flows(g, m)
    qLm, qGm = opening_flows_model(m, st["h"])
    c, n, r = hole_frames(m)
    Zw = c[:, 0] * m.sa + c[:, 2] * m.ca
    near = c[:, 1] <= 0
    sub, exp_ = near & (Zw < st["h"] - 1.5), near & (Zw > st["h"] + 1.5)
    xs = []
    if sub.any():
        xs.append((float(np.median(c[sub, 0])), "submerged opening side"))
    if exp_.any():
        xs.append((float(np.median(c[exp_, 0])), "exposed opening side"))
    for i, (x0, lab) in enumerate(xs):                           # snap to the nearest opening
        k = np.argmin(np.abs(c[:, 0] - x0) + 1e3 * ~(sub if "submerged" in lab else exp_))
        xs[i] = (float(c[k, 0]), lab)
    if xs:
        sections(g, st, xs, head, os.path.join(out, "sections.png"))
    openings_figure(m, st, qLc, qGc, qLm, qGm, os.path.join(out, "openings.png"),
                    f"{label}, pool level {st['h']:g} mm: flow through every opening (t = {avg_from:.3g}–{t_end:.3g} s)")
    try:
        view3d(g, m, st, qLc, qGc, os.path.join(out, "view3d.png"),
               f"{label}, outlet end {TILT:g} deg up, pool {st['h']:g} mm, t = {t_end:.3g} s: pressure on the "
               "drain, liquid surface (blue), flow through each opening")
    except Exception as e:                                       # noqa: BLE001
        print("  3D view skipped:", e)

    if args.frames:
        fdir = os.path.join(out, "frames")
        os.makedirs(fdir, exist_ok=True)
        import imageio.v2 as imageio  # noqa: E402
        imgs = []
        for f in files:
            gi = load(f)
            ti = file_time(gi, f, mon)
            pth = os.path.join(fdir, f"midplane_{ti:.4f}.png")
            plane_world(gi, frame, st, (0, 1, 0), (0, 0, 0),
                        f"{label}, pool {st['h']:g} mm, Kratos two-fluid CFD, t = {ti:.3f} s", pth,
                        ((-8, 140), (-3.5, 36)), False)
            imgs.append(imageio.imread(pth))
        hmin = min(i.shape[0] for i in imgs)
        wmin = min(i.shape[1] for i in imgs)
        imageio.mimsave(os.path.join(out, "midplane.gif"), [i[:hmin, :wmin] for i in imgs], duration=0.25)

    summary = dict(model=st, cfd=dict(t_end=t_end, avg_from=avg_from, **hist,
                                      qL_openings_ml_min=float(np.sum(qLc) * 6e7),
                                      qG_openings_l_min=float(np.sum(qGc) * 6e4)),
                   per_opening=dict(x=m.pattern.x.tolist(), theta=m.pattern.theta.tolist(),
                                    qL_cfd=(qLc * 6e7).tolist(), qG_cfd=(qGc * 6e4).tolist(),
                                    qL_model=(qLm * 6e7).tolist(), qG_model=(qGm * 6e4).tolist()))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary["cfd"], indent=1))


if __name__ == "__main__":
    main()

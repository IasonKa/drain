"""Pictures and numbers of a fixed-level OpenFOAM run (vof/local/runs/<name>), next to the model.

    py -3.12 vof/local/render_cfd.py <run dir> [--avg 0.04]

Reads only text files written by the run: the sampled planes (postProcessing/slices/<t>/*.vtk,
legacy ASCII), the monitors (postProcessing/*/0/*.dat) and, for the flow through every opening,
the cell centres, volumes, U and alpha.liquid of the last write (ASCII fields).
Writes into <run>/pictures/:
  midplane.png      liquid/air, pressure, velocity in the plane through the axis, tilted as in the pool
  outlet_end.png    close-up of the outlet end: the inner liquid level, the bore and the air jets
  row.png           the plane through a row of openings: jets through the openings
  sections.png      cross-sections through a submerged and an exposed opening
  openings.png      flow rate through every opening, CFD and model, on the unrolled skin
  history.png       outlet liquid and air flow and the pressure inside the drain against time
  view3d.png        3D: pressure on the drain, the liquid surface, flow through each opening (PyVista)
  summary.json      averaged CFD numbers and the model's
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOF = os.path.dirname(HERE)
ROOT = os.path.dirname(VOF)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_study  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.edgecolor": "#5f5e5a", "axes.labelcolor": "#2c2c2a", "axes.titlesize": 9.5,
    "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a", "xtick.labelsize": 8, "ytick.labelsize": 8,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
INK, MUTED = "#2c2c2a", "#898781"
PHASE = mcolors.LinearSegmentedColormap.from_list("phase", ["#fbfaf6", "#9ec8ef", "#2a78d6"])
VEL_NORM = mcolors.LogNorm(1e-3, 40)
SPECS = {"usz_tilt10": dict(layout="usz", d=1.0, family="reference", tilt=10.0), "rec_tilt10": None}


# ----------------------------------------------------------------------------- reading
def read_vtk(path):
    """Legacy ASCII POLYDATA written by OpenFOAM: points (m), polygons and CELL_DATA fields."""
    tok = open(path).read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    pts = np.array(tok[i + 3: i + 3 + 3 * n], float).reshape(n, 3)
    i = tok.index("POLYGONS")
    nf, size = int(tok[i + 1]), int(tok[i + 2])
    raw = np.array(tok[i + 3: i + 3 + size], int)
    faces, k = [], 0
    for _ in range(nf):
        c = raw[k]
        faces.append(raw[k + 1: k + 1 + c])
        k += c + 1
    fields = {}
    if "CELL_DATA" in tok:
        i = tok.index("CELL_DATA")
        j = i + 2
        if tok[j] == "FIELD":
            nfld = int(tok[j + 2])
            j += 3
            for _ in range(nfld):
                name, ncomp, nt = tok[j], int(tok[j + 1]), int(tok[j + 2])
                vals = np.array(tok[j + 4: j + 4 + ncomp * nt], float).reshape(nt, ncomp)
                fields[name] = vals[:, 0] if ncomp == 1 else vals
                j += 4 + ncomp * nt
        else:                                       # SCALARS / VECTORS blocks
            while j < len(tok):
                kind, name = tok[j], tok[j + 1]
                if kind == "SCALARS":
                    j += 3 if tok[j + 3] != "LOOKUP_TABLE" else 0
                    j = tok.index("LOOKUP_TABLE", j) + 2
                    fields[name] = np.array(tok[j: j + nf], float)
                    j += nf
                elif kind == "VECTORS":
                    fields[name] = np.array(tok[j + 3: j + 3 + 3 * nf], float).reshape(nf, 3)
                    j += 3 + 3 * nf
                else:
                    break
    return pts, faces, fields


def read_field(path):
    """internalField of an ASCII OpenFOAM volScalarField / volVectorField."""
    txt = open(path).read()
    m = re.search(r"internalField\s+(uniform|nonuniform)", txt)
    if m.group(1) == "uniform":
        raise ValueError(f"{path}: uniform field")
    m2 = re.search(r"nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", txt[m.start():])
    n = int(m2.group(2))
    start = m.start() + m2.end()
    end = txt.index("\n)", start)
    block = txt[start:end].replace("(", " ").replace(")", " ")
    arr = np.array(block.split(), float)
    return arr.reshape(n, 3) if m2.group(1) == "vector" else arr


def read_dat(path):
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    return np.array([[float(v) for v in r] for r in rows])


def monitor(run, name):
    files = sorted(glob.glob(os.path.join(run, "postProcessing", name, "*", "*.dat")),
                   key=lambda p: float(os.path.basename(os.path.dirname(p))))
    data = [read_dat(f) for f in files]
    data = [d for d in data if d.size]
    if not data:
        return None
    d = np.concatenate(data)
    _, idx = np.unique(d[:, 0], return_index=True)
    return d[idx]


def times(run):
    out = []
    for d in os.listdir(run):
        try:
            t = float(d)
        except ValueError:
            continue
        if t > 0 and os.path.exists(os.path.join(run, d, "U")):
            out.append((t, d))
    return sorted(out)


# ----------------------------------------------------------------------------- geometry
class Frame:
    def __init__(self, tilt):
        al = math.radians(tilt)
        self.sa, self.ca = math.sin(al), math.cos(al)

    def world(self, x, z):            # drain frame (mm) -> world (X across, Z up), mm
        return x * self.ca - z * self.sa, x * self.sa + z * self.ca

    def world_vec(self, ux, uz):
        return ux * self.ca - uz * self.sa, ux * self.sa + uz * self.ca


def plane_coords(pts, normal, origin, up):
    """2D coordinates in a plane: e1 = up x normal ... returns (u, v) in mm."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    v_dir = np.asarray(up, float) - np.dot(up, n) * n
    v_dir /= np.linalg.norm(v_dir)
    u_dir = np.cross(v_dir, n)
    d = (pts - np.asarray(origin)) * 1e3
    return d @ u_dir, d @ v_dir, u_dir, v_dir


# ----------------------------------------------------------------------------- slices
def slice_avg(run, name, tsel):
    """Average a sampled plane over the selected write times (same mesh at every time)."""
    acc, pts, faces = None, None, None
    for _, d in tsel:
        p = glob.glob(os.path.join(run, "postProcessing", "slices", d, f"{name}*.vtk"))
        if not p:
            continue
        pts, faces, f = read_vtk(p[0])
        if acc is None:
            acc = {k: v.copy() for k, v in f.items()}
            n = 1
        else:
            for k in acc:
                acc[k] += f[k]
            n += 1
    if acc is None:
        return None
    return pts, faces, {k: v / n for k, v in acc.items()}


def draw_field(ax, polys, values, cmap, norm, lw=0.0):
    pc = PolyCollection(polys, array=values, cmap=cmap, norm=norm, edgecolors="face", linewidths=lw)
    ax.add_collection(pc)
    return pc


def streamlines(ax, cx, cz, ux, uz, alpha, xlim, zlim, n=420, color="#1f1f1f"):
    """Streamlines of the in-plane velocity, interpolated from the face centres to a grid."""
    tri = mtri.Triangulation(cx, cz)
    a, b, c = tri.triangles.T
    edge = np.max([np.hypot(cx[a] - cx[b], cz[a] - cz[b]), np.hypot(cx[b] - cx[c], cz[b] - cz[c]),
                   np.hypot(cx[c] - cx[a], cz[c] - cz[a])], axis=0)
    tri.set_mask(edge > 1.2)                       # no triangles across the drain wall
    gx = np.linspace(*xlim, n)
    gz = np.linspace(*zlim, int(n * (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])) + 2)
    GX, GZ = np.meshgrid(gx, gz)
    iu = mtri.LinearTriInterpolator(tri, ux)(GX, GZ)
    iv = mtri.LinearTriInterpolator(tri, uz)(GX, GZ)
    speed = np.hypot(iu, iv)
    ax.streamplot(gx, gz, iu.filled(np.nan), iv.filled(np.nan), density=2.2, color=color,
                  linewidth=0.45, arrowsize=0.55, broken_streamlines=True)
    return speed


def plane_figure(run, st, frame, name, tsel, title, path, crop=None, stream=True):
    s = slice_avg(run, name, tsel)
    if s is None:
        return None
    pts, faces, f = s
    pm = pts * 1e3
    X, Z = frame.world(pm[:, 0], pm[:, 2])
    polys = [np.column_stack([X[fc], Z[fc]]) for fc in faces]
    cx = np.array([p[:, 0].mean() for p in polys])
    cz = np.array([p[:, 1].mean() for p in polys])
    U = f["U"]
    uX, uZ = frame.world_vec(U[:, 0], U[:, 2])
    spd = np.linalg.norm(U, axis=1)
    if crop:
        keep = (cx > crop[0][0] - 2) & (cx < crop[0][1] + 2) & (cz > crop[1][0] - 2) & (cz < crop[1][1] + 2)
    else:
        keep = np.ones(cx.size, bool)
    polys = [p for p, k in zip(polys, keep) if k]
    cx, cz, uX, uZ, spd = cx[keep], cz[keep], uX[keep], uZ[keep], spd[keep]
    alpha, p_ = f["alpha.liquid"][keep], f["p"][keep]
    xlim = crop[0] if crop else (cx.min(), cx.max())
    zlim = crop[1] if crop else (cz.min(), cz.max())
    asp = (zlim[1] - zlim[0]) / (xlim[1] - xlim[0])
    W = 12.0
    fig, axes = plt.subplots(3, 1, figsize=(W, 3 * W * asp * 0.86 + 1.4), constrained_layout=True)
    pr = max(abs(np.percentile(p_, 1)), abs(np.percentile(p_, 99)), 50)
    specs = [("liquid fraction α (blue liquid, white air)", alpha, PHASE, mcolors.Normalize(0, 1)),
             ("pressure p (Pa, gauge)", p_, "RdBu_r", mcolors.TwoSlopeNorm(0, -pr, max(pr * 0.4, 30))),
             ("velocity magnitude |U| (m/s, log scale)", np.clip(spd, 1e-3, 40), "magma", VEL_NORM)]
    for ax, (lab, val, cmap, norm) in zip(axes, specs):
        pc = draw_field(ax, polys, val, cmap, norm)
        cb = fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.01, aspect=18)
        cb.set_label(lab, fontsize=8)
        if lab.startswith("velocity") and stream:
            streamlines(ax, cx, cz, uX, uZ, alpha, xlim, zlim, color="#d6f5ff")
        ax.axhline(st["h"], color="#2a78d6", lw=0.7, ls=(0, (4, 3)))
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal")
        ax.set_facecolor("#7b7a74")                 # solid wall shows through
        ax.set_ylabel("height (mm)")
    axes[-1].set_xlabel("horizontal position (mm)")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return dict(alpha=alpha, p=p_, cx=cx, cz=cz)


def section_figure(run, st, tsel, names, path, title):
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 3.6 * len(names)), constrained_layout=True)
    for r, (name, lab) in enumerate(names):
        s = slice_avg(run, name, tsel)
        if s is None:
            continue
        pts, faces, f = s
        pm = pts * 1e3
        polys = [np.column_stack([pm[fc, 1], pm[fc, 2]]) for fc in faces]
        cy = np.array([p[:, 0].mean() for p in polys])
        cz = np.array([p[:, 1].mean() for p in polys])
        U, alpha, p_ = f["U"], f["alpha.liquid"], f["p"]
        spd = np.linalg.norm(U, axis=1)
        x0 = pm[:, 0].mean()
        pr = max(abs(np.percentile(p_, 1)), abs(np.percentile(p_, 99)), 50)
        for c, (clab, val, cmap, norm) in enumerate([
                ("liquid fraction α", alpha, PHASE, mcolors.Normalize(0, 1)),
                ("pressure p (Pa)", p_, "RdBu_r", mcolors.TwoSlopeNorm(0, -pr, max(pr * 0.4, 30))),
                ("|U| (m/s), arrows: in-plane velocity", np.clip(spd, 1e-3, 40), "magma", VEL_NORM)]):
            ax = axes[r, c] if len(names) > 1 else axes[c]
            pc = draw_field(ax, polys, val, cmap, norm)
            fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.01).set_label(clab, fontsize=8)
            if c == 2:
                sel = np.where((np.abs(cy) < 11) & (cz > -2.5) & (cz < 12))[0]
                sel = sel[np.argsort(spd[sel])[::-1][:900]]
                vv = U[sel][:, 1:] / np.maximum(spd[sel], 1e-9)[:, None]
                ax.quiver(cy[sel], cz[sel], vv[:, 0], vv[:, 1], color="#d6f5ff", scale=60, width=0.0022,
                          headwidth=3.2)
            ax.set_xlim(-11, 11)
            ax.set_ylim(-2.5, 12)
            ax.set_aspect("equal")
            ax.set_facecolor("#7b7a74")
            ax.set_title(f"{lab}, x = {x0:.1f} mm from the tip", fontsize=8.5)
            ax.set_xlabel("y (mm)")
            ax.set_ylabel("z, drain frame (mm)")
    fig.suptitle(title, fontsize=10.5, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- openings
def opening_flows_cfd(run, m, t_dir):
    """Liquid and air flow (m3/s, positive into the drain) through every opening, from the cells
    inside the opening's bore: Q = (1/t) * sum(-U.n * V) over the cells within the wall slab."""
    cdir = os.path.join(run, t_dir)
    for fld, fo in (("C", "writeCellCentres"), ("V", "writeCellVolumes")):
        if not os.path.exists(os.path.join(cdir, fld)):
            subprocess.run(["postProcess", "-func", fo, "-time", t_dir], cwd=run,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    C = read_field(os.path.join(cdir, "C")) * 1e3
    V = read_field(os.path.join(cdir, "V"))
    U = read_field(os.path.join(cdir, "U"))
    al = np.clip(read_field(os.path.join(cdir, "alpha.liquid")), 0, 1)
    pat, dr = m.pattern, m.drain
    a, b, t = dr.a, dr.b, dr.wall
    qL, qG = np.zeros(pat.n), np.zeros(pat.n)
    near = (np.abs(C[:, 1]) < a + 0.5) & (C[:, 2] > -0.5) & (C[:, 2] < 2 * b + 0.5)
    Ci, Vi, Ui, ali = C[near], V[near], U[near], al[near]
    for k in range(pat.n):
        th = pat.theta[k]
        ny, nz = math.sin(th) / a, -math.cos(th) / b
        nn = math.hypot(ny, nz)
        n = np.array([0.0, ny / nn, nz / nn])                     # outward normal
        p_out = np.array([pat.x[k], a * math.sin(th), b * (1 - math.cos(th))])
        p_mid = p_out - 0.5 * t * n
        r = 0.5 * pat.d[k] * math.sqrt(pat.area_scale[k])
        d = Ci - p_mid
        ax_ = d @ n
        rad = np.linalg.norm(d - ax_[:, None] * n, axis=1)
        sel = (np.abs(ax_) < 0.5 * t) & (rad < r)
        if not sel.any():
            continue
        flux = -(Ui[sel] @ n) * Vi[sel] / (t * 1e-3)             # m3/s
        qL[k] = np.sum(flux * ali[sel])
        qG[k] = np.sum(flux * (1 - ali[sel]))
    return qL, qG


def opening_flows_model(m, h):
    from make_pictures import opening_flows       # noqa: E402  (scripts/make_pictures.py)
    br = np.zeros(m.pattern.n, bool)
    for hh in np.arange(m.h_start(), h, -0.25):
        br = m.state(hh + m.Z_min, br)["breached"]
    H = h + m.Z_min
    st = m.state(H, br)
    qL, qG, qX = opening_flows(m, H, st)
    return qL - qX, qG


def openings_figure(m, st, qLc, qGc, qLm, qGm, path, title):
    pat = m.pattern
    s = m.ell.s_of(pat.theta)
    P = m.ell.perimeter
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.2), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    qmax_l = max(np.max(np.abs(qLc)), np.max(np.abs(qLm)), 1e-12) * 6e7
    qmax_g = max(np.max(qGc), np.max(qGm), 1e-12) * 6e4
    for c, (lab, qL, qG) in enumerate((("CFD (OpenFOAM interFoam)", qLc, qGc), ("model (network)", qLm, qGm))):
        for r, (kind, q, scale, unit, cmap) in enumerate((
                ("liquid in", qL * 6e7, qmax_l, "mL/min", "Blues"),
                ("air in", qG * 6e4, qmax_g, "L/min", "Oranges"))):
            ax = axes[r, c]
            sc = ax.scatter(pat.x, s, c=np.clip(q, 0, None), cmap=cmap, vmin=0, vmax=scale,
                            s=30 * (pat.d / 1.0) ** 2, edgecolors="#5f5e5a", linewidths=0.4)
            cb = fig.colorbar(sc, ax=ax, pad=0.01)
            cb.set_label(f"{kind} per opening ({unit})", fontsize=8)
            ax.set_xlim(-2, m.drain.length + 2)
            ax.set_ylim(-P / 2, P / 2)
            ax.set_yticks([-P / 2, -P / 4, 0, P / 4, P / 2])
            ax.set_yticklabels(["top", "−y side", "bottom", "+y side", "top"])
            ax.set_title(f"{lab}: {kind}, total {np.sum(np.clip(q, 0, None)):.0f} {unit}"
                         if unit == "mL/min" else f"{lab}: {kind}, total {np.sum(np.clip(q, 0, None)):.1f} {unit}",
                         fontsize=9)
            ax.set_xlabel("x from the tip (mm)")
            ax.grid(color="#e1e0d9", lw=0.5)
            # waterline on the skin (world level h)
            th = np.linspace(-np.pi, np.pi, 721)
            zl = m.drain.b * (1 - np.cos(th))
            xw = (st["h"] + m.Z_min - zl * m.ca) / m.sa
            ok = (xw > 0) & (xw < m.drain.length)
            ax.plot(xw[ok], m.ell.s_of(th[ok]), ".", ms=0.8, color="#2a78d6")
    fig.suptitle(title + "\nunrolled skin: each dot is one opening; the blue dotted line is the waterline",
                 fontsize=10, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ----------------------------------------------------------------------------- 3D (PyVista)
def render3d(run, st, m, t_dir, qLc, qGc, path, title):
    """Drain surface coloured by pressure, liquid surface (alpha = 0.5) and every opening as a dot
    coloured by its flow rate, in the world frame (outlet end raised), seen from below the side."""
    try:
        import pyvista as pv
    except ImportError:
        print("  pyvista not installed - 3D view skipped")
        return
    pv.OFF_SCREEN = True
    pdir = os.path.join(run, "postProcessing")
    drain_f = glob.glob(os.path.join(pdir, "slices", t_dir, "drain*.vtk"))
    fs_f = glob.glob(os.path.join(pdir, "freeSurface", t_dir, "*.vtk"))
    if not drain_f:
        return
    tilt = st["tilt"]

    def prep(mesh):
        mesh = mesh.scale(1e3, inplace=False)
        mesh.rotate_y(-tilt, inplace=True)
        return mesh
    drain = prep(pv.read(drain_f[0]))
    pl = pv.Plotter(off_screen=True, window_size=(2200, 1150))
    pl.set_background("white")
    pl.add_mesh(drain, scalars="p", cmap="RdBu_r", clim=(-950, 950), smooth_shading=False,
                scalar_bar_args=dict(title="pressure on the drain (Pa)", vertical=True, position_x=0.9,
                                     position_y=0.55, height=0.38, width=0.035, title_font_size=18,
                                     label_font_size=15, color="black", fmt="%.0f"))
    if fs_f:
        fs = prep(pv.read(fs_f[0]))
        pl.add_mesh(fs, color="#4f97e0", opacity=0.35, smooth_shading=True)
    pat, dr = m.pattern, m.drain
    xo = pat.x
    yo = dr.a * np.sin(pat.theta)
    zo = dr.b * (1 - np.cos(pat.theta))
    nyo, nzo = np.sin(pat.theta) / dr.a, -np.cos(pat.theta) / dr.b
    nn = np.hypot(nyo, nzo)
    pts = np.column_stack([xo, yo + 0.35 * nyo / nn, zo + 0.35 * nzo / nn])
    al = math.radians(tilt)
    Xw = pts[:, 0] * math.cos(al) - pts[:, 2] * math.sin(al)
    Zw = pts[:, 0] * math.sin(al) + pts[:, 2] * math.cos(al)
    P = np.column_stack([Xw, pts[:, 1], Zw])
    liq = qLc * 6e7
    air = qGc * 6e4
    for mask, vals, cmap, lab, fmt in ((liq > 0.05, liq, "Blues", "liquid in per opening (mL/min)", "%.0f"),
                                       (air > 0.01, air, "Oranges", "air in per opening (L/min)", "%.2f")):
        if mask.any():
            cloud = pv.PolyData(P[mask])
            cloud["q"] = vals[mask]
            pl.add_mesh(cloud, scalars="q", cmap=cmap, clim=(0, max(vals[mask].max(), 1e-6)),
                        point_size=17, render_points_as_spheres=True,
                        scalar_bar_args=dict(title=lab, vertical=True,
                                             position_x=0.9 if cmap == "Blues" else 0.74, position_y=0.08,
                                             height=0.38, width=0.035, title_font_size=18, label_font_size=15,
                                             color="black", fmt=fmt))
    pl.add_text(title, position="upper_left", font_size=12, color="black")
    pl.camera_position = [(55, -125, -125), (62, 0, 12), (0, 0, 1)]    # from below the near side
    pl.camera.zoom(1.12)
    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(path)
    pl.close()


# ----------------------------------------------------------------------------- history
def history(run, st, path, title, avg_from):
    of, ol = monitor(run, "outletFlow"), monitor(run, "outletLiquid")
    lp = monitor(run, "lumenProbes")
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    out = {}
    if of is not None and ol is not None:
        n = min(len(of), len(ol))
        t = of[:n, 0]
        ql = ol[:n, 1] * 6e7
        qg = (of[:n, 1] - ol[:n, 1]) * 6e4
        axes[0].plot(t, ql, color="#2a78d6", lw=1)
        axes[0].axhline(st["qL_ml_min"], color="#2a78d6", ls="--", lw=1, label="model")
        axes[1].plot(t, qg, color="#eb6834", lw=1)
        axes[1].axhline(st["qG_l_min"], color="#eb6834", ls="--", lw=1, label="model")
        w = t >= avg_from
        out.update(qL_ml_min=float(np.mean(ql[w])) if w.any() else None,
                   qG_l_min=float(np.mean(qg[w])) if w.any() else None)
    if lp is not None:
        # probes: (p alpha) per location; the gas pressure is p at the drain-top probes in air
        t = lp[:, 0]
        nloc = (lp.shape[1] - 1) // 2
        p = lp[:, 1: 1 + nloc]
        a = lp[:, 1 + nloc: 1 + 2 * nloc]
        pg = np.array([np.mean(pr[ar < 0.5]) if np.any(ar < 0.5) else np.nan
                       for pr, ar in zip(p[:, :5], a[:, :5])])
        axes[2].plot(t, pg, color="#6250d6", lw=1, label="CFD, top of the drain interior (in air)")
        axes[2].axhline(st["pg"], color="#6250d6", ls="--", lw=1, label="model gas pressure")
        w = (t >= avg_from) & np.isfinite(pg)
        out["pg"] = float(np.mean(pg[w])) if w.any() else None
    axes[0].set_ylabel("liquid out\n(mL/min)")
    axes[1].set_ylabel("air out\n(L/min)")
    axes[2].set_ylabel("pressure inside\nthe drain (Pa)")
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(color="#e1e0d9", lw=0.5)
        ax.axvspan(avg_from, ax.get_xlim()[1], color="#f1efe8", zorder=0)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle(title + "\n(shaded: averaging window)", fontsize=10, color=INK)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--avg", type=float, default=0.04, help="s at the end used for averages")
    args = ap.parse_args()
    run = os.path.abspath(args.run)
    st = json.load(open(os.path.join(run, "model_state.json")))
    case = st["case"]
    spec = SPECS[case] or dict(run_study.best_spec(), tilt=st["tilt"])
    m = run_study.make_model(spec)
    frame = Frame(st["tilt"])
    out_dir = os.path.join(run, "pictures")
    os.makedirs(out_dir, exist_ok=True)
    ts = times(run)
    if not ts:
        sys.exit("no field writes yet")
    t_end = ts[-1][0]
    avg_from = max(t_end - args.avg, ts[0][0])
    tsel = [(t, d) for t, d in ts if t >= avg_from - 1e-9]
    label = "USZ as built" if case.startswith("usz") else "Recommended"
    head = (f"{label}, outlet end {st['tilt']:g}° up, pool level {st['h']:g} mm, suction −7 mmHg — "
            f"OpenFOAM interFoam, average of t = {tsel[0][0]:.3g}–{t_end:.3g} s")

    hist = history(run, st, os.path.join(out_dir, "history.png"), head, avg_from)
    plane_figure(run, st, frame, "midPlane", tsel, head + "\nplane through the drain axis",
                 os.path.join(out_dir, "midplane.png"), crop=((-8, 140), (-3.5, 36)), stream=False)
    plane_figure(run, st, frame, "midPlane", tsel, head + "\noutlet end, plane through the axis",
                 os.path.join(out_dir, "outlet_end.png"), crop=((92, 136), (12, 34)))
    plane_figure(run, st, frame, "rowPlane", tsel, head + "\nplane through a row of openings",
                 os.path.join(out_dir, "row.png"), crop=((-8, 140), (-3.5, 36)), stream=False)
    section_figure(run, st, tsel, [("xLiquid", "submerged opening"), ("xAir", "exposed opening")],
                   os.path.join(out_dir, "sections.png"), head)

    qLc, qGc = opening_flows_cfd(run, m, ts[-1][1])
    qLm, qGm = opening_flows_model(m, st["h"])
    openings_figure(m, st, qLc, qGc, qLm, qGm, os.path.join(out_dir, "openings.png"),
                    f"{label}, pool level {st['h']:g} mm, t = {t_end:.3g} s: flow through every opening")
    try:                                           # needs OpenGL; skip on a headless machine
        render3d(run, st, m, ts[-1][1], qLc, qGc, os.path.join(out_dir, "view3d.png"),
                 f"{label}, outlet end {st['tilt']:g} deg up, pool {st['h']:g} mm, t = {t_end:.3g} s: "
                 "pressure on the drain, liquid surface (blue), flow through each opening")
    except Exception as e:                         # noqa: BLE001
        print("  3D view skipped:", e)
    summary = dict(model=st, cfd=dict(t_end=t_end, avg_from=avg_from, **hist,
                                      qL_openings_ml_min=float(np.sum(qLc) * 6e7),
                                      qG_openings_l_min=float(np.sum(qGc) * 6e4)),
                   per_opening=dict(x=m.pattern.x.tolist(), theta=m.pattern.theta.tolist(),
                                    qL_cfd=(qLc * 6e7).tolist(), qG_cfd=(qGc * 6e4).tolist(),
                                    qL_model=(qLm * 6e7).tolist(), qG_model=(qGm * 6e4).tolist()))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps(summary["cfd"], indent=1))


if __name__ == "__main__":
    main()

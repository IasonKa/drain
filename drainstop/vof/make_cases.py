"""Build the 3D two-phase (OpenFOAM interFoam) check of the drainstop stop levels.

Two cases, both with the outlet end tilted 10 deg up:

    usz_tilt10   as-built USZ hole field, usz_as_built() in drainstop/model.py     model stop 20.0 mm
    rec_tilt10   recommended layout, results/recommended.json (custom_two_rows)   model stop  1.4 mm

For each case this writes vof/cases/<case>/:
  - the drain wall as one closed STL surface, built here with numpy from the analytic geometry
    (elliptic skin, closed tip, end face with the bore, short outlet tube, every opening drilled
    normal to the skin), plus helper surfaces for the pool end wall and the refinement regions;
  - blockMesh + snappyHexMesh dictionaries (coarse and fine), topoSet, setFields (full and short
    start) and interFoam dictionaries with the monitors that the run scripts and analyse_run.py read;
  - case.json (geometry, predictions, probe lines, pool volume table) and case.env (for Allmesh).
It also writes vof/runtime_estimate.md: cell counts from an octree estimate of the snappyHexMesh
refinement, physical time and time step from the reduced-order model.

    py -3.12 vof/make_cases.py                  # both cases + runtime estimate (about 1 min)
    py -3.12 vof/make_cases.py --no-estimate    # cases only

Frame: everything is built in the drain frame (x along the drain from the tip, y lateral, z up from
the bottom line when lying flat; origin at the bottom of the tip). The tilt enters only through
gravity, g = -9.81 (sin a, 0, cos a), so the mesh stays aligned with the drain. The world height
above the lowest point of the skin, h = x sin a + z cos a, is the pool level h of the model.
Lengths are in mm in this file and in metres in the OpenFOAM case.
"""
import argparse
import json
import math
import os
import shutil
import stat
import struct
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from drainstop.model import Drain, DrainModel, Ellipse, Fluid, Gas, Suction  # noqa: E402
from run_study import build  # noqa: E402

TILT = 10.0
REC = json.load(open(os.path.join(ROOT, "results", "recommended.json")))
CASES = {
    "usz_tilt10": dict(label="USZ as built (318 x 1.0 mm drilled through both faces)",
                       spec=dict(layout="usz", tilt=TILT)),
    "rec_tilt10": dict(label="Recommended: two rows at +-30 deg, staggered, 36 x 1.0 mm",
                       spec=dict(REC, tilt=TILT)),
}

GEO = dict(
    gap_side=5.0,     # mm, pool beside the drain (at least; rounded up to the background grid)
    gap_tip=5.0,      # mm, pool beyond the tip
    gap_end=4.0,      # mm, pool beyond the outlet end face, closed by a wall the outlet tube pierces
    gap_floor=2.5,    # mm, floor below the bottom line, parallel to the drain
    air_above=3.0,    # mm, air above the initial level at the lowest corner of the domain top
    t_tip=0.3,        # mm, tip cap (thin: the USZ field has openings 0.4 mm from the tip)
    t_end=0.4,        # mm, outlet end face = bore length (USZ openings reach 118.4 mm)
    tube_wall=0.8,    # mm, outlet tube wall
    tube_d=4.0,       # outlet tube length beyond the end face, in bore diameters (about)
    base=1.6,         # mm, background (blockMesh) cell
    n_ring=480,       # points around the skin ends, the end faces and the tube
    rim_dx=0.1,       # mm, point spacing on the opening rims
    skin_ds=0.3,      # mm, background spacing on the skin, around
    skin_dx=1.0,      # mm, background spacing on the skin, along (the skin is straight along x)
)

# refinement levels above the background cell: level n -> base / 2^n (1.6, 0.8, 0.4, 0.2, 0.1 mm)
RES = {
    "coarse": dict(pool=1, lumen=2, skin=2, holes=3, bore=3),
    "fine":   dict(pool=1, lumen=2, skin=3, holes=4, bore=4),
}
FEATURE_BAND = 0.2        # mm, cells this close to an opening rim get the 'holes' level (with the
                          # buffer cells this covers the whole Ø1 mm opening and its rims)
N_BUFFER = 2              # snappyHexMesh nCellsBetweenLevels
MAX_CO = 1.0              # maxCo and maxAlphaCo (interFoam tutorial values)
Q_STOP_ML_MIN = 10.0      # mL/min, "the pool has stopped draining" (about 1 % of the full-bore flow)
STOP_WINDOW_S = 0.3       # s, window over which the pool drain rate is averaged for the stop test
SHORT_ABOVE = 5.0         # mm, the short start begins this far above the model's stop level
COST_US = 10.0            # us per cell, time step and core for interFoam (assumed; calibrate on Euler)


# ============================================================================ geometry
class Body:
    """Elliptic drain skin: outer y = a sin(th), z = b (1 - cos(th)); inner = outer offset by the wall."""

    def __init__(self, drain):
        self.a, self.b, self.t, self.L = drain.a, drain.b, drain.wall, drain.length
        self.rp = drain.port_d / 2
        self.ell = Ellipse(drain.a, drain.b)
        self.P = self.ell.perimeter

    def outer(self, th):
        return self.a * np.sin(th), self.b * (1 - np.cos(th))

    def normal(self, th):
        ny, nz = np.sin(th) / self.a, -np.cos(th) / self.b
        m = np.hypot(ny, nz)
        return ny / m, nz / m

    def inner(self, th):
        y, z = self.outer(th)
        ny, nz = self.normal(th)
        return y - self.t * ny, z - self.t * nz

    def theta_outer(self, y, z):
        return np.arctan2(y / self.a, 1 - z / self.b)

    def ring_thetas(self, n):
        """Parameters of n points at equal polar angle about the axis (0, b), on the outer and the
        inner skin. Point 0 is on the top line (th = -pi); the angle runs with th."""
        phi = np.pi / 2 + 2 * np.pi * np.arange(n) / n
        th_o = np.arctan2(np.cos(phi) / self.a, -np.sin(phi) / self.b)
        th_o[0] = -np.pi
        th_o = np.where(th_o > np.pi - 1e-12, th_o - 2 * np.pi, th_o)
        tt = np.linspace(-np.pi, np.pi, 200001)
        y, z = self.inner(tt)
        ph = np.unwrap(np.arctan2(z - self.b, y))
        th_i = np.interp(phi, ph, tt)
        th_i[0] = -np.pi
        return phi, th_o, th_i


def hole_rims(body, pat, rim_dx):
    """Outer and inner rims of every opening: a cylinder of the opening's area drilled along the skin
    normal at its centre, cut with the outer skin and with the inner (offset) skin.

    Returns a list of dicts with 3D rim points (mm) and their developed coordinates (s, x), where s is
    the outer arc length of the skin parameter (inner points use the parameter of the inner skin).
    """
    out = []
    for i in range(pat.n):
        r = 0.5 * pat.d[i] * math.sqrt(pat.area_scale[i])
        n = max(16, int(math.ceil(2 * math.pi * r / rim_dx)))
        th_c, x_c = float(pat.theta[i]), float(pat.x[i])
        yc, zc = body.outer(th_c)
        ny, nz = body.normal(th_c)
        ty, tz = body.a * math.cos(th_c), body.b * math.sin(th_c)
        m = math.hypot(ty, tz)
        ty, tz = ty / m, tz / m
        ph = 2 * np.pi * np.arange(n) / n
        px = x_c + r * np.cos(ph)
        py = yc + r * np.sin(ph) * ty
        pz = zc + r * np.sin(ph) * tz
        # outer skin: ((py + l ny)/a)^2 + ((pz + l nz - b)/b)^2 = 1, root nearest l = 0
        A = ny ** 2 / body.a ** 2 + nz ** 2 / body.b ** 2
        B = 2 * (py * ny / body.a ** 2 + (pz - body.b) * nz / body.b ** 2)
        C = py ** 2 / body.a ** 2 + (pz - body.b) ** 2 / body.b ** 2 - 1
        disc = np.sqrt(B * B - 4 * A * C)
        l1, l2 = (-B + disc) / (2 * A), (-B - disc) / (2 * A)
        lo = np.where(np.abs(l1) < np.abs(l2), l1, l2)
        oy, oz = py + lo * ny, pz + lo * nz
        th_o = body.theta_outer(oy, oz)
        # inner skin: parameter th with inner(th) on the drill line, by bracketing and bisection
        tt = th_c + np.linspace(-0.6, 0.6, 601)
        iy, iz = body.inner(tt[None, :])
        g = (iy - py[:, None]) * nz - (iz - pz[:, None]) * ny
        sg = np.sign(g)
        ch = sg[:, :-1] * sg[:, 1:] <= 0
        k = np.where(ch, np.abs(np.arange(600) - 299.5), np.inf).argmin(1)
        lo_t, hi_t = tt[k], tt[k + 1]

        def gf(t_):
            y_, z_ = body.inner(t_)
            return (y_ - py) * nz - (z_ - pz) * ny
        g_lo = gf(lo_t)
        for _ in range(50):
            mid = 0.5 * (lo_t + hi_t)
            g_m = gf(mid)
            left = np.sign(g_m) == np.sign(g_lo)
            lo_t, g_lo = np.where(left, mid, lo_t), np.where(left, g_m, g_lo)
            hi_t = np.where(left, hi_t, mid)
        th_i = 0.5 * (lo_t + hi_t)
        iy, iz = body.inner(th_i)
        s_o, s_i = body.ell.s_of(th_o), body.ell.s_of(th_i)
        for s in (s_o, s_i):
            if s.max() - s.min() > body.P / 2 or np.abs(s).max() > body.P / 2 - 0.05:
                raise ValueError(f"opening {i} crosses the top line; move the seam")
        out.append(dict(r=r, x=x_c, theta=th_c, n=(0.0, ny, nz), c=(x_c, yc, zc),
                        outer=np.column_stack([px, oy, oz]), inner=np.column_stack([px, iy, iz]),
                        sx_outer=np.column_stack([s_o, px]), sx_inner=np.column_stack([s_i, px])))
    return out


def _delaunay(pts):
    try:
        from scipy.spatial import Delaunay
        return Delaunay(pts).simplices
    except ImportError:
        import matplotlib.tri as mtri
        return mtri.Triangulation(pts[:, 0], pts[:, 1]).triangles


def triangulate_skin(P, x_lo, x_hi, edge_s, rims, ds, dx, seed):
    """Delaunay triangulation of the developed skin [-P/2, P/2] x [x_lo, x_hi] with the openings cut out.

    edge_s: s of the ring points on the two x edges (ring order; edge_s[0] = -P/2 is the top line).
    rims:   list of (n_k, 2) rim polygons in (s, x).
    Every boundary segment is kept free of other points (ghost points outside, background points
    held back from edges and rims), so the boundary comes out exactly as given; triangles with a
    ghost corner, and triangles spanned by the rim of a single opening, are dropped.
    Returns the points, the triangles and index blocks for the edges, the seam and the rims.
    """
    n_ring = len(edge_s)
    e_s = np.append(edge_s, P / 2)
    e_ring = np.append(np.arange(n_ring), 0)
    n_seam = int(math.ceil((x_hi - x_lo) / 0.2)) + 1
    x_seam = np.linspace(x_lo, x_hi, n_seam)[1:-1]
    blocks, pts = {}, []

    def add(name, p):
        start = sum(len(q) for q in pts)
        pts.append(np.asarray(p, float))
        blocks[name] = np.arange(start, start + len(p))

    add("lo", np.column_stack([e_s, np.full(len(e_s), x_lo)]))
    add("hi", np.column_stack([e_s, np.full(len(e_s), x_hi)]))
    add("seamL", np.column_stack([np.full(len(x_seam), -P / 2), x_seam]))
    add("seamR", np.column_stack([np.full(len(x_seam), P / 2), x_seam]))
    for k, rim in enumerate(rims):
        add(f"rim{k}", rim)
    # background grid, jittered, kept away from edges, seam and rims
    rng = np.random.default_rng(seed)
    gs = np.arange(-P / 2 + ds / 2, P / 2, ds)
    gx = np.arange(x_lo + dx / 2, x_hi, dx)
    S, X = np.meshgrid(gs, gx, indexing="ij")
    S = S + rng.uniform(-0.15, 0.15, S.shape) * ds
    X = X + rng.uniform(-0.15, 0.15, X.shape) * dx
    S, X = S.ravel(), X.ravel()
    ok = (np.abs(np.abs(S) - P / 2) > 0.2) & (X - x_lo > 0.15) & (x_hi - X > 0.15)
    for rim in rims:
        c = rim.mean(0)
        R = np.hypot(*(rim - c).T).max() + 0.15
        near = np.abs(X - c[1]) < R
        ok[near] &= np.hypot(S[near] - c[0], X[near] - c[1]) > R
    add("bg", np.column_stack([S[ok], X[ok]]))
    # ghost points outside the rectangle
    g = []
    for off in (0.4, 1.2):
        ss = np.arange(-P / 2 - off, P / 2 + off + 1e-9, 0.4)
        xx = np.arange(x_lo - off, x_hi + off + 1e-9, 0.4)
        g += [np.column_stack([ss, np.full(len(ss), x_lo - off)]), np.column_stack([ss, np.full(len(ss), x_hi + off)]),
              np.column_stack([np.full(len(xx), -P / 2 - off), xx]), np.column_stack([np.full(len(xx), P / 2 + off), xx])]
    add("ghost", np.concatenate(g))
    pts = np.concatenate(pts)
    tri = _delaunay(pts)
    owner = np.full(len(pts), -1)
    for k in range(len(rims)):
        owner[blocks[f"rim{k}"]] = k
    ghost = np.zeros(len(pts), bool)
    ghost[blocks["ghost"]] = True
    drop = ghost[tri].any(1)
    o = owner[tri]
    drop |= (o[:, 0] >= 0) & (o[:, 0] == o[:, 1]) & (o[:, 1] == o[:, 2])
    tri = tri[~drop]
    # the boundary must be exactly the rectangle and the rims
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    expect = (len(blocks["lo"]) - 1) * 2 + (len(x_seam) + 1) * 2 + sum(len(r) for r in rims)
    if cnt.max() > 2 or (cnt == 1).sum() != expect:
        raise RuntimeError(f"skin triangulation: {(cnt == 1).sum()} boundary edges, expected {expect}")
    blocks["lo_ring"], blocks["hi_ring"] = e_ring, e_ring
    return pts, tri, blocks


class Mesh:
    """Indexed triangle surface collected from parts (mm)."""

    def __init__(self):
        self.V, self.F, self.nv = [], [], 0

    def verts(self, p):
        p = np.asarray(p, float).reshape(-1, 3)
        idx = np.arange(self.nv, self.nv + len(p))
        self.V.append(p)
        self.nv += len(p)
        return idx

    def faces(self, f, want):
        """Add faces (m, 3), flipped where their normal opposes want(centroid, normal) < 0."""
        f = np.asarray(f, int).reshape(-1, 3)
        V = np.concatenate(self.V)
        a, b, c = V[f[:, 0]], V[f[:, 1]], V[f[:, 2]]
        nrm = np.cross(b - a, c - a)
        flip = np.einsum("ij,ij->i", nrm, want((a + b + c) / 3)) < 0
        f[flip] = f[flip][:, ::-1]
        self.F.append(f)

    def arrays(self):
        return np.concatenate(self.V), np.concatenate(self.F)


def strip(mesh, ring_a, ring_b, want, closed=True):
    """Quads between two rings of vertex indices with equal length."""
    n = len(ring_a)
    j = np.arange(n if closed else n - 1)
    k = (j + 1) % n
    f = np.concatenate([np.column_stack([ring_a[j], ring_a[k], ring_b[k]]),
                        np.column_stack([ring_a[j], ring_b[k], ring_b[j]])])
    mesh.faces(f, want)


def disc(mesh, ring, centre, fracs, want):
    """Planar star-shaped face: rings shrunk towards the centre, then a fan."""
    V = np.concatenate(mesh.V)
    pts = V[ring]
    prev = ring
    for fr in fracs:
        cur = mesh.verts(centre + fr * (pts - centre))
        strip(mesh, prev, cur, want)
        prev = cur
    c = mesh.verts(centre)[0]
    n = len(prev)
    j = np.arange(n)
    mesh.faces(np.column_stack([prev[j], prev[(j + 1) % n], np.full(n, c)]), want)


def blend(mesh, ring_a, ring_b, fracs, want):
    """Planar face between two star-shaped rings with matching points."""
    V = np.concatenate(mesh.V)
    pa, pb = V[ring_a], V[ring_b]
    prev = ring_a
    for fr in fracs:
        cur = mesh.verts((1 - fr) * pa + fr * pb)
        strip(mesh, prev, cur, want)
        prev = cur
    strip(mesh, prev, ring_b, want)


def drain_surface(body, rims, dom):
    """Closed surface of the drain wall: outer skin, inner skin, opening walls, tip cap, end face with
    the bore and the outlet tube (its end lies beyond the domain, where snappyHexMesh cuts it)."""
    a, b, t, L = body.a, body.b, body.t, body.L
    tt, te = GEO["t_tip"], GEO["t_end"]
    ri, ro = body.rp, body.rp + GEO["tube_wall"]
    x_te = dom["x1"] + 1.0
    phi, th_o, th_i = body.ring_thetas(GEO["n_ring"])
    m = Mesh()
    ex = np.array([1.0, 0, 0])

    def ring_pts(x, yz):
        return np.column_stack([np.full(len(yz[0]), x), yz[0], yz[1]])

    def circle(x, r):
        return np.column_stack([np.full(len(phi), x), r * np.cos(phi), b + r * np.sin(phi)])

    out_lo = m.verts(ring_pts(0.0, body.outer(th_o)))
    out_hi = m.verts(ring_pts(L, body.outer(th_o)))
    in_lo = m.verts(ring_pts(tt, body.inner(th_i)))
    in_hi = m.verts(ring_pts(L - te, body.inner(th_i)))

    def grad(c):
        g = np.zeros_like(c)
        g[:, 1], g[:, 2] = c[:, 1] / a ** 2, (c[:, 2] - b) / b ** 2
        return g

    # opening rims (shared by skin and opening wall)
    rim_o = [m.verts(h["outer"]) for h in rims]
    rim_i = [m.verts(h["inner"]) for h in rims]
    for side, x_lo, x_hi, ring_th, ring_lo, ring_hi, rim_v, key, sign, mapper in (
            ("outer", 0.0, L, th_o, out_lo, out_hi, rim_o, "sx_outer", 1.0, body.outer),
            ("inner", tt, L - te, th_i, in_lo, in_hi, rim_i, "sx_inner", -1.0, body.inner)):
        edge_s = body.ell.s_of(ring_th)
        edge_s[0] = -body.P / 2
        pts, tri, blk = triangulate_skin(body.P, x_lo, x_hi, edge_s, [h[key] for h in rims],
                                         GEO["skin_ds"], GEO["skin_dx"], seed=7 if side == "outer" else 8)
        gid = np.full(len(pts), -1)
        gid[blk["lo"]] = ring_lo[blk["lo_ring"]]
        gid[blk["hi"]] = ring_hi[blk["hi_ring"]]
        seam_x = pts[blk["seamL"], 1]
        sy, sz = mapper(np.full(len(seam_x), np.pi))
        seam = m.verts(np.column_stack([seam_x, sy, sz]))
        gid[blk["seamL"]], gid[blk["seamR"]] = seam, seam
        for k in range(len(rims)):
            gid[blk[f"rim{k}"]] = rim_v[k]
        bg = pts[blk["bg"]]
        yz = mapper(body.ell.theta(bg[:, 0]))
        gid[blk["bg"]] = m.verts(np.column_stack([bg[:, 1], yz[0], yz[1]]))
        m.faces(gid[tri], lambda c, s=sign: s * grad(c))

    # opening walls, facing the opening axis
    for h, ro_v, ri_v in zip(rims, rim_o, rim_i):
        C, n = np.array(h["c"]), np.array(h["n"])

        def towards_axis(c, C=C, n=n):
            v = c - C
            return -(v - np.outer(v @ n, n))
        strip(m, ro_v, ri_v, towards_axis)

    # tip cap, end faces and the outlet tube
    disc(m, out_lo, np.array([0.0, 0, b]), (0.8, 0.6, 0.4, 0.2), lambda c: -ex + 0 * c)
    disc(m, in_lo, np.array([tt, 0, b]), (0.8, 0.6, 0.4, 0.2), lambda c: ex + 0 * c)
    tube_o0 = m.verts(circle(L, ro))
    blend(m, out_hi, tube_o0, (0.25, 0.5, 0.75), lambda c: ex + 0 * c)
    bore0 = m.verts(circle(L - te, ri))
    blend(m, in_hi, bore0, (0.25, 0.5, 0.75), lambda c: -ex + 0 * c)

    def radial(c, s):
        g = np.zeros_like(c)
        g[:, 1], g[:, 2] = c[:, 1], c[:, 2] - b
        return s * g
    xs_o = np.linspace(L, x_te, int(math.ceil((x_te - L) / 0.5)) + 1)
    prev = tube_o0
    for x in xs_o[1:]:
        cur = m.verts(circle(x, ro))
        strip(m, prev, cur, lambda c: radial(c, 1.0))
        prev = cur
    tube_oe = prev
    xs_i = np.linspace(L - te, x_te, int(math.ceil((x_te - L + te) / 0.5)) + 1)
    prev = bore0
    for x in xs_i[1:]:
        cur = m.verts(circle(x, ri))
        strip(m, prev, cur, lambda c: radial(c, -1.0))
        prev = cur
    strip(m, prev, tube_oe, lambda c: ex + 0 * c)
    return m


def check_closed(V, F, n_handles, name):
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    used = np.unique(F)
    chi = len(used) - len(u) + len(F)
    ok = cnt.min() == 2 and cnt.max() == 2 and chi == 2 - 2 * n_handles
    # consistent orientation: every edge is used once in each direction
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    ud = np.unique(d, axis=0)
    ok &= len(ud) == len(d)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    print(f"  {name}: {len(used)} vertices, {len(F)} triangles, edges ok={cnt.min() == cnt.max() == 2}, "
          f"Euler char {chi} (expected {2 - 2 * n_handles}), oriented={len(ud) == len(d)}, volume {vol:.1f} mm3")
    if not ok:
        raise RuntimeError(f"{name} is not a closed, oriented surface")
    return vol


def box_surface(lo, hi):
    """Closed axis-aligned box with outward normals (8 vertices, 12 triangles)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    F = np.array([f for q in quads for f in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    return V, F


def prism_surface(yz, x0, x1, fan_centre):
    """Closed straight prism over a star-shaped section (n, 2), outward normals."""
    m = Mesh()
    r0 = m.verts(np.column_stack([np.full(len(yz), x0), yz]))
    r1 = m.verts(np.column_stack([np.full(len(yz), x1), yz]))
    yc, zc = fan_centre

    def out(c):
        g = np.zeros_like(c)
        g[:, 1], g[:, 2] = c[:, 1] - yc, c[:, 2] - zc
        return g
    strip(m, r0, r1, out)
    ex = np.array([1.0, 0, 0])
    disc(m, r0, np.array([x0, yc, zc]), (), lambda c: -ex + 0 * c)
    disc(m, r1, np.array([x1, yc, zc]), (), lambda c: ex + 0 * c)
    return m.arrays()


def pool_end_surface(dom, b, r_hole):
    """Pool end wall: plane x = x_end over the whole section, with a hole inside the tube wall."""
    xw = dom["x_end"]
    y0, y1, z0, z1 = dom["y0"] - 3, dom["y1"] + 3, dom["z0"] - 3, dom["z1"] + 3
    corners = np.array([[y1, z1], [y0, z1], [y0, z0], [y1, z0]])
    ang = np.linspace(0, 2 * np.pi, 257)[:-1]
    ang = np.sort(np.concatenate([ang, np.mod(np.arctan2(corners[:, 1] - b, corners[:, 0]), 2 * np.pi)]))
    d = np.column_stack([np.cos(ang), np.sin(ang)])
    with np.errstate(divide="ignore"):
        ty = np.where(d[:, 0] > 0, y1 / d[:, 0], y0 / d[:, 0])
        tz = np.where(d[:, 1] > 0, (z1 - b) / d[:, 1], (z0 - b) / d[:, 1])
    tr = np.minimum(np.abs(ty), np.abs(tz))
    m = Mesh()
    inner = m.verts(np.column_stack([np.full(len(ang), xw), r_hole * d[:, 0], b + r_hole * d[:, 1]]))
    outer = m.verts(np.column_stack([np.full(len(ang), xw), tr * d[:, 0], b + tr * d[:, 1]]))
    ex = np.array([1.0, 0, 0])
    blend(m, inner, outer, (0.1, 0.25, 0.5), lambda c: -ex + 0 * c)
    return m.arrays()


def pool_region_surface(dom, h_top, sa, ca):
    """Closed region of the trough below world height h_top (refined to the 'pool' level)."""
    x0, x1 = dom["x0"] - 1, dom["x_end"] + 1
    y0, y1, z0 = dom["y0"] - 1, dom["y1"] + 1, dom["z0"] - 1
    zt0, zt1 = (h_top - x0 * sa) / ca, (h_top - x1 * sa) / ca
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, zt0], [x1, y0, zt1], [x1, y1, zt1], [x0, y1, zt0]], float)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    F = np.array([f for q in quads for f in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))])
    return V, F


def write_stl(path, V, F, scale=1e-3):
    """Binary STL in metres."""
    P = (V * scale).astype(np.float32)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = n, np.stack([a, b, c], 1)
    with open(path, "wb") as f:
        f.write(b"binary STL, drainstop vof/make_cases.py".ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())


# ============================================================================ domain and pool
def domain(body, sa, ca, h0_full):
    """Trough around the drain in the drain frame, sized on the background grid."""
    h = GEO["base"]
    L, a, b = body.L, body.a, body.b
    gap_end = GEO["gap_end"]
    k = max(1, round((GEO["tube_d"] * 2 * body.rp - gap_end) / h))
    x1 = L + gap_end + k * h                       # outlet = domain end; pool end wall on a grid plane
    nx = int(math.ceil((x1 + GEO["gap_tip"]) / h - 1e-9))
    ny = int(math.ceil(2 * (a + GEO["gap_side"]) / h - 1e-9))
    z0 = -GEO["gap_floor"]
    x0 = x1 - nx * h
    z_top = (h0_full + GEO["air_above"] - x0 * sa) / ca
    nz = int(math.ceil((z_top - z0) / h - 1e-9))
    return dict(x0=x0, x1=x1, y0=-ny * h / 2, y1=ny * h / 2, z0=z0, z1=z0 + nz * h, nx=nx, ny=ny, nz=nz,
                x_end=L + gap_end, l_tube=x1 - L)


def pool_table(body, dom, sa, ca, dh=0.05):
    """Liquid volume of the pool (outside the drain and the tube) below world height h, by x-slices."""
    a, b, L = body.a, body.b, body.L
    ro = body.rp + GEO["tube_wall"]
    W = dom["y1"] - dom["y0"]
    x = np.linspace(dom["x0"], dom["x_end"], 6001)
    dx = x[1] - x[0]
    h_lo = dom["x0"] * sa + dom["z0"] * ca
    hs = np.arange(h_lo, dom["x_end"] * sa + dom["z1"] * ca, dh)

    def seg(u):          # fraction of an ellipse/circle area below a chord at u * semi-axis
        u = np.clip(u, -1, 1)
        return (np.pi - np.arccos(u) + u * np.sqrt(1 - u * u)) / np.pi

    V = []
    for H in hs:
        zl = (H - x * sa) / ca
        area = W * np.clip(zl - dom["z0"], 0, dom["z1"] - dom["z0"])
        body_ = (x >= 0) & (x <= L)
        area = area - np.where(body_, np.pi * a * b * seg((zl - b) / b), 0.0)
        tube = x > L
        area = area - np.where(tube, np.pi * ro * ro * seg((zl - b) / ro), 0.0)
        V.append(area.sum() * dx)
    return hs, np.array(V)


def free_surface_area(hs, V):
    return np.gradient(V, hs)


def level_lines(body, dom, sa, ca):
    """World-vertical probe lines in the pool, from the floor up (drain frame, mm)."""
    a = body.a
    yg = 0.5 * (a + dom["y1"])
    starts = [(-0.5 * (0 - dom["x0"]), 0.0)]
    starts += [(x, s * yg) for x in (5.0, 15.0, 35.0, 60.0, 85.0, 110.0) for s in (1, -1)]
    u = np.array([sa, 0.0, ca])
    ds = 0.4
    lines = []
    for x, y in starts:
        floor = np.array([x, y, dom["z0"]])
        p0 = floor + 0.5 * ds * u          # probe i samples the slab [i ds, (i+1) ds] above the floor
        s_max = min((dom["z1"] - 0.3 - p0[2]) / ca, (dom["x_end"] - 0.3 - p0[0]) / sa)
        n = int(s_max // ds) + 1
        pts = p0[None, :] + (np.arange(n) * ds)[:, None] * u[None, :]
        lines.append(dict(start=p0.tolist(), h_floor=float(floor[0] * sa + floor[2] * ca), ds=ds, n=n,
                          points=pts))
    return lines


# ============================================================================ OpenFOAM dictionaries
HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
  drainstop VOF check, written by vof/make_cases.py
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _w(case, rel, cls, body, obj=None):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="\n") as f:
        f.write(HDR.format(cls=cls, obj=obj or os.path.basename(rel)) + textwrap.dedent(body).lstrip("\n"))


def _script(case, rel, body):
    p = os.path.join(case, rel)
    with open(p, "w", newline="\n") as f:
        f.write(textwrap.dedent(body).lstrip("\n"))
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def v3(p, s=1e-3):
    return "({:.9g} {:.9g} {:.9g})".format(*(np.asarray(p, float) * s))


def max_delta_t(res, fluid, gas):
    """Half the capillary time-step limit sqrt(rho_avg dx^3 / (2 pi sigma)) of the smallest cell."""
    lev = max(RES[res]["holes"], RES[res]["bore"])
    dx = GEO["base"] / 2 ** lev * 1e-3
    lim = math.sqrt(0.5 * (fluid.rho + gas.rho) * dx ** 3 / (2 * math.pi * fluid.gamma))
    return float(f"{0.5 * lim:.2g}")


def write_case(name, info, body, rims, dom, model, fluid, gas, suction, sa, ca, out_dir):
    case = os.path.join(out_dir, name)
    if os.path.isdir(case):
        shutil.rmtree(case)
    tri = os.path.join(case, "constant", "triSurface")
    os.makedirs(tri)
    b, L, a, t = body.b, body.L, body.a, body.t
    ri, ro = body.rp, body.rp + GEO["tube_wall"]
    tt, te = GEO["t_tip"], GEO["t_end"]
    x1 = dom["x1"]

    # ---------------------------------------------------------------- surfaces
    print(f"{name}: {model.pattern.n} openings")
    m = drain_surface(body, rims, dom)
    V, F = m.arrays()
    check_closed(V, F, model.pattern.n, "drain.stl")
    write_stl(os.path.join(tri, "drain.stl"), V, F)
    V, F = pool_end_surface(dom, b, 0.5 * (ri + ro))
    write_stl(os.path.join(tri, "poolEnd.stl"), V, F)
    V, F = pool_region_surface(dom, info["h0_full"] + 1.0, sa, ca)
    check_closed(V, F, 0, "poolRegion.stl")
    write_stl(os.path.join(tri, "poolRegion.stl"), V, F)
    phi, th_o, th_i = body.ring_thetas(192)
    V, F = prism_surface(np.column_stack(body.inner(th_i)), tt, L - te, (0.0, b))
    check_closed(V, F, 0, "lumen.stl")
    write_stl(os.path.join(tri, "lumen.stl"), V, F)
    V, F = prism_surface(np.column_stack(body.outer(th_o)), 0.0, L, (0.0, b))
    check_closed(V, F, 0, "envelope.stl")
    write_stl(os.path.join(tri, "envelope.stl"), V, F)

    # ---------------------------------------------------------------- case.json / case.env
    hs, Vp = info["pool_h"], info["pool_V"]
    lines = level_lines(body, dom, sa, ca)
    lumen_pts = [(x, 0.0, 2 * b - t - 1.0) for x in (15.0, 40.0, 65.0, 90.0, 112.0)]
    lumen_pts += [(L - te - 1.5, 0.0, b + 1.5)]
    tube_plane_x = L + 2 * 2 * ri
    md = {k: max_delta_t(k, fluid, gas) for k in RES}
    end_time = {k: info[f"t_end_{k}"] for k in ("full", "short")}
    case_json = dict(
        name=name, label=info["label"], tilt_deg=TILT, spec=info["spec"],
        model=dict(h_stop=info["h_stop"], lip=info["lip"], h_top=info["h_top"],
                   h_stop_simple=info["h_ana_simple"], q_port_ml_min=info["q_port"] * 6e7),
        start=dict(full=info["h0_full"], short=info["h0_short"]),
        end_time=end_time, max_delta_t=md, q_stop_ml_min=Q_STOP_ML_MIN, stop_window_s=STOP_WINDOW_S,
        fluid=dict(rho=fluid.rho, mu=fluid.mu, sigma=fluid.gamma), gas=dict(rho=gas.rho, mu=gas.mu),
        p_outlet=suction.p_set, gravity=[-9.81 * sa, 0.0, -9.81 * ca],
        geometry=dict(GEO, a=a, b=b, length=L, wall=t, port_d=2 * ri, tube_ro=ro, domain=dom),
        level_lines=[dict(start=ln["start"], h_floor=ln["h_floor"], ds=ln["ds"], n=ln["n"]) for ln in lines],
        lumen_probes=lumen_pts, tube_plane_x=tube_plane_x,
        pool_table=dict(h=np.round(hs, 4).tolist(), V_mm3=np.round(Vp, 2).tolist()),
    )
    json.dump(case_json, open(os.path.join(case, "case.json"), "w"), indent=1)
    with open(os.path.join(case, "case.env"), "w", newline="\n") as f:
        f.write(textwrap.dedent(f"""\
            # written by vof/make_cases.py, read by Allmesh
            H0_full={info['h0_full']:.3f}       # mm, initial pool level, full start (3 mm above the drain top)
            H0_short={info['h0_short']:.3f}      # mm, initial pool level, short start
            END_TIME_full={end_time['full']:g}      # s, upper limit (the watcher stops earlier)
            END_TIME_short={end_time['short']:g}
            WRITE_full=0.1         # s, field output interval
            WRITE_short=0.05
            MAX_DT_coarse={md['coarse']:g}   # s, half the capillary time-step limit of the smallest cell
            MAX_DT_fine={md['fine']:g}
            RHO_AIR={gas.rho:g}
            MU_AIR={gas.mu:g}
            """))

    # ---------------------------------------------------------------- constant
    _w(case, "constant/g", "uniformDimensionedVectorField", f"""
    // gravity in the drain frame: outlet end tilted {TILT:g} deg up
    dimensions      [0 1 -2 0 0 0 0];
    value           ({-9.81 * sa:.6f} 0 {-9.81 * ca:.6f});
    """)
    _w(case, "constant/transportProperties", "dictionary", f"""
    phases          (liquid air);

    liquid          // {fluid.name}
    {{
        transportModel  Newtonian;
        nu              {fluid.mu / fluid.rho:.6g};
        rho             {fluid.rho:g};
    }}

    air
    {{
        transportModel  Newtonian;
        nu              {gas.mu / gas.rho:.6g};
        rho             {gas.rho:g};
    }}

    sigma           {fluid.gamma:g};
    """)
    _w(case, "constant/turbulenceProperties", "dictionary", """
    simulationType  laminar;
    """)

    # ---------------------------------------------------------------- 0.orig
    walls = "\"(walls|drain|poolEnd)\""
    _w(case, "0.orig/U", "volVectorField", f"""
    dimensions      [0 1 -1 0 0 0 0];
    internalField   uniform (0 0 0);
    boundaryField
    {{
        {walls}
        {{
            type            noSlip;
        }}
        atmosphere
        {{
            type            pressureInletOutletVelocity;
            value           uniform (0 0 0);
        }}
        outlet
        {{
            type            pressureInletOutletVelocity;
            value           uniform (0 0 0);
        }}
    }}
    """)
    _w(case, "0.orig/p_rgh", "volScalarField", f"""
    dimensions      [1 -1 -2 0 0 0 0];
    internalField   uniform 0;
    boundaryField
    {{
        {walls}
        {{
            type            fixedFluxPressure;
            value           uniform 0;
        }}
        atmosphere          // open pool surface, atmospheric (gauge 0)
        {{
            type            prghTotalPressure;
            p0              uniform 0;
            value           uniform 0;
        }}
        outlet              // regulated suction, -7 mmHg static in the outlet tube
        {{
            type            prghPressure;
            p               uniform {suction.p_set:.3f};
            value           uniform 0;
        }}
    }}
    """)
    _w(case, "0.orig/alpha.liquid", "volScalarField", f"""
    dimensions      [0 0 0 0 0 0 0];
    internalField   uniform 0;
    boundaryField
    {{
        {walls}
        {{
            type            constantAlphaContactAngle;
            theta0          90;     // as theta_e = 90 deg in the model (no capillary hold)
            limit           gradient;
            value           uniform 0;
        }}
        atmosphere
        {{
            type            inletOutlet;
            inletValue      uniform 0;
            value           uniform 0;
        }}
        outlet
        {{
            type            inletOutlet;
            inletValue      uniform 0;
            value           uniform 0;
        }}
    }}
    """)

    # ---------------------------------------------------------------- system: mesh
    h = GEO["base"]
    lo, hi = (dom["x0"], dom["y0"], dom["z0"]), (dom["x1"], dom["y1"], dom["z1"])
    vx = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
          (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    _w(case, "system/blockMeshDict", "dictionary", f"""
    // trough around the drain, drain frame, {h:g} mm background cells
    scale           1;
    vertices
    (
        {chr(10).join('        ' + v3(v) for v in vx).lstrip()}
    );
    blocks
    (
        hex (0 1 2 3 4 5 6 7) ({dom['nx']} {dom['ny']} {dom['nz']}) simpleGrading (1 1 1)
    );
    edges           ();
    boundary
    (
        walls       // tip end, sides and floor of the pool
        {{
            type    wall;
            faces   ((0 4 7 3) (0 1 5 4) (3 7 6 2) (0 3 2 1));
        }}
        atmosphere  // top, always in air
        {{
            type    patch;
            faces   ((4 5 6 7));
        }}
        outlet      // end of the outlet tube (the rest of this face is cut away)
        {{
            type    patch;
            faces   ((1 2 6 5));
        }}
    );
    """)
    _w(case, "system/surfaceFeatureExtractDict", "dictionary", """
    drain.stl
    {
        extractionMethod    extractFromSurface;
        includedAngle       150;
        extractFromSurfaceCoeffs { includedAngle 150; }
        subsetFeatures
        {
            nonManifoldEdges    no;
            openEdges           yes;
        }
        writeObj            no;
    }
    """)
    loc = (60.37, 0.5 * (a + dom["y1"]) + 0.013, 3.017)
    for res, lv in RES.items():
        _w(case, f"system/snappyHexMeshDict.{res}", "dictionary", obj="snappyHexMeshDict", body=f"""
        // {res}: skin {h / 2 ** lv['skin']:g} mm, openings/caps {h / 2 ** lv['holes']:g} mm, bore {h / 2 ** lv['bore']:g} mm,
        //        lumen {h / 2 ** lv['lumen']:g} mm, pool {h / 2 ** lv['pool']:g} mm, air {h:g} mm
        castellatedMesh true;
        snap            true;
        addLayers       false;

        geometry
        {{
            drain.stl       {{ type triSurfaceMesh; name drain; }}
            poolEnd.stl     {{ type triSurfaceMesh; name poolEnd; }}
            poolRegion.stl  {{ type triSurfaceMesh; name poolRegion; }}
            lumen.stl       {{ type triSurfaceMesh; name lumen; }}
            boreZone        {{ type searchableCylinder; point1 {v3((L - te - 2.0, 0, b))}; point2 {v3((L + 3.0, 0, b))}; radius {(ro + 0.5) * 1e-3:.6g}; }}
            tubeZone        {{ type searchableCylinder; point1 {v3((L - te, 0, b))}; point2 {v3((x1 + 1.0, 0, b))}; radius {ri * 1e-3:.6g}; }}
            tipZone         {{ type searchableBox; min {v3((-0.6, -a - 0.6, -0.6))}; max {v3((tt + 0.6, a + 0.6, 2 * b + 0.6))}; }}
            endZone         {{ type searchableBox; min {v3((L - te - 0.6, -a - 0.6, -0.6))}; max {v3((L + 0.6, a + 0.6, 2 * b + 0.6))}; }}
        }}

        castellatedMeshControls
        {{
            maxLocalCells       3000000;
            maxGlobalCells      40000000;
            minRefinementCells  10;
            maxLoadUnbalance    0.10;
            nCellsBetweenLevels {N_BUFFER};
            features
            (
                {{ file "drain.eMesh"; levels (({FEATURE_BAND * 1e-3:g} {lv['holes']})); }}    // opening rims, caps, bore lip
            );
            refinementSurfaces
            {{
                drain   {{ level ({lv['skin']} {lv['skin']}); patchInfo {{ type wall; }} }}
                poolEnd {{ level (1 1); patchInfo {{ type wall; }} }}
            }}
            resolveFeatureAngle 30;
            refinementRegions
            {{
                poolRegion  {{ mode inside; levels ((1e15 {lv['pool']})); }}    // liquid at the start
                lumen       {{ mode inside; levels ((1e15 {lv['lumen']})); }}
                tubeZone    {{ mode inside; levels ((1e15 {lv['lumen']})); }}
                boreZone    {{ mode inside; levels ((1e15 {lv['bore']})); }}
                tipZone     {{ mode inside; levels ((1e15 {lv['holes']})); }}    // 0.3 mm tip cap
                endZone     {{ mode inside; levels ((1e15 {lv['holes']})); }}    // 0.4 mm end face
            }}
            locationInMesh {v3(loc)};      // pool, beside the drain
            allowFreeStandingZoneFaces true;
        }}

        snapControls
        {{
            nSmoothPatch            3;
            tolerance               2.0;
            nSolveIter              50;
            nRelaxIter              5;
            nFeatureSnapIter        10;
            implicitFeatureSnap     false;
            explicitFeatureSnap     true;
            multiRegionFeatureSnap  false;
        }}

        addLayersControls
        {{
            relativeSizes           true;
            layers                  {{}}
            expansionRatio          1.0;
            finalLayerThickness     0.3;
            minThickness            0.1;
            nGrow                   0;
            featureAngle            60;
            slipFeatureAngle        30;
            nRelaxIter              3;
            nSmoothSurfaceNormals   1;
            nSmoothNormals          3;
            nSmoothThickness        10;
            maxFaceThicknessRatio   0.5;
            maxThicknessToMedialRatio 0.3;
            minMedialAxisAngle      90;
            nBufferCellsNoExtrude   0;
            nLayerIter              50;
        }}

        meshQualityControls
        {{
            maxNonOrtho             65;
            maxBoundarySkewness     20;
            maxInternalSkewness     4;
            maxConcave              80;
            minVol                  1e-30;      // m3: cells here are down to 1e-12 m3
            minTetQuality           1e-30;
            minArea                 -1;
            minTwist                0.02;
            minDeterminant          0.001;
            minFaceWeight           0.05;
            minVolRatio             0.01;
            minTriangleTwist        -1;
            nSmoothScale            4;
            errorReduction          0.75;
            relaxed
            {{
                maxNonOrtho         75;
            }}
        }}

        mergeTolerance  1e-6;
        """)

    # lumen cell zone: cells inside the outer envelope of the drain, plus the outlet tube
    _w(case, "system/topoSetDict", "dictionary", f"""
    // cellZone 'lumen': drain interior (and the openings) + outlet tube. Pool = everything else.
    actions
    (
        {{
            name    lumenSet;
            type    cellSet;
            action  new;
            source  surfaceToCell;
            file    "constant/triSurface/envelope.stl";
            useSurfaceOrientation true;     // closed, outward oriented
            outsidePoints   ({v3(loc)});
            includeCut      false;
            includeInside   true;
            includeOutside  false;
            nearDistance    -1;
            curvature       -100;
        }}
        {{
            name    lumenSet;
            type    cellSet;
            action  add;
            source  cylinderToCell;
            point1  {v3((L, 0, b))};
            point2  {v3((x1 + 1.0, 0, b))};
            p1      {v3((L, 0, b))};
            p2      {v3((x1 + 1.0, 0, b))};
            radius  {0.5 * (ri + ro) * 1e-3:.6g};
        }}
        {{
            name    lumen;
            type    cellZoneSet;
            action  new;
            source  setToCellZone;
            set     lumenSet;
        }}
    );
    """)

    # initial liquid: everything below a horizontal level h0 (pool, drain interior and tube alike)
    u = np.array([sa, 0, ca])
    e = np.array([ca, 0, -sa])
    for start in ("full", "short"):
        h0 = info[f"h0_{start}"]
        D, E0, E, Y = 80.0, -20.0, 180.0, 60.0
        origin = (h0 - D) * u + E0 * e + np.array([0, -Y / 2, 0])
        _w(case, f"system/setFieldsDict.{start}", "dictionary", obj="setFieldsDict", body=f"""
        // {start} start: liquid below the world level h0 = {h0:.2f} mm above the lowest point of the skin,
        // the same level inside and outside the drain. Box edges i (horizontal), j (lateral), k (up).
        defaultFieldValues
        (
            volScalarFieldValue alpha.liquid 0
        );
        regions
        (
            rotatedBoxToCell
            {{
                origin  {v3(origin)};
                i       {v3(E * e)};
                j       {v3((0, Y, 0))};
                k       {v3(D * u)};
                fieldValues
                (
                    volScalarFieldValue alpha.liquid 1
                );
            }}
        );
        """)

    # ---------------------------------------------------------------- system: solver
    probe_txt = "\n".join("            " + v3(p) for ln in lines for p in ln["points"])
    lumen_txt = "\n".join("            " + v3(p) for p in lumen_pts)
    _w(case, "system/controlDict", "dictionary", f"""
    application     interFoam;
    startFrom       latestTime;
    startTime       0;
    stopAt          endTime;            // the watcher sets writeNow when the pool has stopped
    endTime         {end_time['full']:g};    // Allmesh sets it from case.env
    deltaT          1e-06;
    writeControl    adjustable;
    writeInterval   0.1;
    purgeWrite      0;
    writeFormat     binary;
    writePrecision  8;
    writeCompression off;
    timeFormat      general;
    timePrecision   8;
    runTimeModifiable yes;
    adjustTimeStep  yes;
    maxCo           {MAX_CO:g};
    maxAlphaCo      {MAX_CO:g};
    maxDeltaT       {md['coarse']:g};       // Allmesh sets it from case.env

    functions
    {{
        // liquid volume in the whole domain and in the drain (+ tube): pool = all - lumen
        liquidVolume
        {{
            type            volFieldValue;
            libs            ("libfieldFunctionObjects.so");
            regionType      all;
            operation       volIntegrate;
            fields          (alpha.liquid);
            writeFields     false;
            writeControl    runTime;
            writeInterval   0.001;
            log             false;
        }}
        lumenVolume
        {{
            $liquidVolume;
            regionType      cellZone;
            name            lumen;
        }}
        // flow at the outlet (m3/s, positive out): all, and liquid only
        outletFlow
        {{
            type            surfaceFieldValue;
            libs            ("libfieldFunctionObjects.so");
            regionType      patch;
            name            outlet;
            operation       sum;
            fields          (phi);
            writeFields     false;
            writeControl    runTime;
            writeInterval   0.001;
            log             false;
        }}
        outletLiquid
        {{
            $outletFlow;
            operation       weightedSum;
            weightField     alpha.liquid;
        }}
        // static pressure and liquid fraction across the tube, two bore diameters downstream
        tubeSection
        {{
            type            surfaceFieldValue;
            libs            ("libfieldFunctionObjects.so");
            regionType      sampledSurface;
            name            tubeSection;
            sampledSurfaceDict
            {{
                type            cuttingPlane;
                planeType       pointAndNormal;
                pointAndNormalDict
                {{
                    point   {v3((tube_plane_x, 0, b))};
                    normal  (1 0 0);
                }}
                interpolate     false;
            }}
            operation       areaAverage;
            fields          (p alpha.liquid);
            writeFields     false;
            writeControl    runTime;
            writeInterval   0.001;
            log             false;
        }}
        // pool level: liquid fraction on {len(lines)} world-vertical lines in the pool ({lines[0]['ds']:g} mm apart)
        levelProbes
        {{
            type            probes;
            libs            ("libsampling.so");
            writeControl    runTime;
            writeInterval   0.005;
            fields          (alpha.liquid);
            fixedLocations  true;
            interpolationScheme cell;
            probeLocations
            (
{probe_txt}
            );
        }}
        // gas pressure in the drain (top of the lumen) and at the bore entrance
        lumenProbes
        {{
            type            probes;
            libs            ("libsampling.so");
            writeControl    runTime;
            writeInterval   0.001;
            fields          (p alpha.liquid);
            fixedLocations  true;
            interpolationScheme cell;
            probeLocations
            (
{lumen_txt}
            );
        }}
        // free surface for ParaView (VTK)
        freeSurface
        {{
            type            surfaces;
            libs            ("libsampling.so");
            writeControl    runTime;
            writeInterval   0.05;
            surfaceFormat   vtk;
            fields          (p U);
            interpolationScheme cellPoint;
            surfaces
            {{
                alpha05
                {{
                    type        isoSurface;
                    isoField    alpha.liquid;
                    isoValue    0.5;
                    interpolate true;
                }}
            }}
        }}
    }}
    """)
    _w(case, "system/fvSchemes", "dictionary", """
    ddtSchemes          { default Euler; }
    gradSchemes         { default Gauss linear; }
    divSchemes
    {
        div(rhoPhi,U)       Gauss linearUpwind grad(U);
        div(phi,alpha)      Gauss vanLeer;
        div(phirb,alpha)    Gauss linear;
        div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
    }
    laplacianSchemes    { default Gauss linear limited corrected 0.5; }
    interpolationSchemes { default linear; }
    snGradSchemes       { default limited corrected 0.5; }
    """)
    _w(case, "system/fvSolution", "dictionary", """
    solvers
    {
        "alpha.liquid.*"
        {
            nAlphaCorr      2;
            nAlphaSubCycles 1;
            cAlpha          1;
            MULESCorr       yes;
            nLimiterIter    5;
            solver          smoothSolver;
            smoother        symGaussSeidel;
            tolerance       1e-8;
            relTol          0;
        }
        "pcorr.*"
        {
            solver          PCG;
            preconditioner  DIC;
            tolerance       1e-5;
            relTol          0;
        }
        p_rgh
        {
            solver          GAMG;
            smoother        DIC;
            tolerance       1e-7;
            relTol          0.05;
        }
        p_rghFinal
        {
            $p_rgh;
            relTol          0;
        }
        "U.*"
        {
            solver          smoothSolver;
            smoother        symGaussSeidel;
            tolerance       1e-6;
            relTol          0;
        }
    }

    PIMPLE
    {
        momentumPredictor           no;
        nOuterCorrectors            1;
        nCorrectors                 3;
        nNonOrthogonalCorrectors    1;
    }

    relaxationFactors
    {
        equations { ".*" 1; }
    }
    """)
    _w(case, "system/decomposeParDict", "dictionary", """
    numberOfSubdomains 48;      // Allmesh sets it to NP
    method          scotch;
    """)

    # ---------------------------------------------------------------- scripts
    _script(case, "Allmesh", """
    #!/bin/bash
    # Mesh and initial fields of one run, in a copy of this case directory.
    #   RES=coarse|fine  START=full|short  NP=<tasks>  [AIR_RHO_FACTOR=1]  ./Allmesh
    # Writes MESHED when the case is decomposed and ready for: mpirun -np $NP interFoam -parallel
    cd "$(dirname "$0")"
    set -e
    source ./case.env
    RES=${RES:-coarse}; START=${START:-full}; NP=${NP:-1}; AIR_RHO_FACTOR=${AIR_RHO_FACTOR:-1}
    eval MAX_DT=\\$MAX_DT_$RES
    eval END_TIME=\\$END_TIME_$START
    eval WRITE=\\$WRITE_$START
    cp system/snappyHexMeshDict.$RES system/snappyHexMeshDict
    cp system/setFieldsDict.$START system/setFieldsDict
    foamDictionary -entry maxDeltaT -set "$MAX_DT" system/controlDict > /dev/null
    foamDictionary -entry endTime -set "$END_TIME" system/controlDict > /dev/null
    foamDictionary -entry writeInterval -set "$WRITE" system/controlDict > /dev/null
    foamDictionary -entry numberOfSubdomains -set "$NP" system/decomposeParDict > /dev/null
    if [ "$AIR_RHO_FACTOR" != "1" ]; then       # heavier air: same stop level, larger time step
        RHO=$(awk -v f="$AIR_RHO_FACTOR" -v r="$RHO_AIR" 'BEGIN{printf "%.6g", r*f}')
        NU=$(awk -v f="$AIR_RHO_FACTOR" -v r="$RHO_AIR" -v m="$MU_AIR" 'BEGIN{printf "%.6g", m/(r*f)}')
        foamDictionary -entry air/rho -set "$RHO" constant/transportProperties > /dev/null
        foamDictionary -entry air/nu -set "$NU" constant/transportProperties > /dev/null
    fi
    echo "Allmesh: RES=$RES START=$START NP=$NP maxDeltaT=$MAX_DT endTime=$END_TIME air x$AIR_RHO_FACTOR"

    blockMesh > log.blockMesh 2>&1
    surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
    if [ "$NP" -gt 1 ]; then
        decomposePar -force > log.decomposePar.mesh 2>&1
        mpirun -np "$NP" snappyHexMesh -parallel -overwrite > log.snappyHexMesh 2>&1
        reconstructParMesh -constant > log.reconstructParMesh 2>&1
        rm -rf processor*
    else
        snappyHexMesh -overwrite > log.snappyHexMesh 2>&1
    fi
    checkMesh -constant > log.checkMesh 2>&1 || true
    grep -E "^ *cells:|Mesh OK|Failed" log.checkMesh || true
    NOUT=$(awk '/^ *outlet *$/{f=1} f && /nFaces/{print $2+0; exit}' constant/polyMesh/boundary)
    if [ "${NOUT:-0}" -eq 0 ]; then
        echo "ERROR: the outlet patch has no faces - the drain interior was cut off from the pool"; exit 2
    fi
    topoSet > log.topoSet 2>&1
    grep -E "lumen" log.topoSet | tail -2 || true
    rm -rf 0 && cp -r 0.orig 0
    setFields > log.setFields 2>&1
    if [ "$NP" -gt 1 ]; then
        decomposePar -force > log.decomposePar 2>&1
    fi
    touch case.foam MESHED
    echo "Allmesh: done"
    """)
    _script(case, "Allclean", """
    #!/bin/bash
    cd "$(dirname "$0")"
    rm -rf processor* postProcessing 0 [1-9]* 0.[0-9]* constant/polyMesh constant/extendedFeatureEdgeMesh \\
           constant/triSurface/*.eMesh log.* MESHED STOPPED WALLTIME FINISHED case.foam analysis \\
           system/snappyHexMeshDict system/setFieldsDict
    """)
    return case


# ============================================================================ estimates
def octree_cells(body, rims, dom, res, h_pool):
    """Approximate snappyHexMesh cell count: refine the background cells by the same rules
    (surface, feature and region levels with nCellsBetweenLevels buffers), count fluid leaves."""
    lv = RES[res]
    h0 = GEO["base"]
    a, b, t, L = body.a, body.b, body.t, body.L
    ai, bi = a - t, b - t
    tt, te = GEO["t_tip"], GEO["t_end"]
    ri, ro = body.rp, body.rp + GEO["tube_wall"]
    al = math.radians(TILT)
    sa, ca = math.sin(al), math.cos(al)
    x_end, x1 = dom["x_end"], dom["x1"]

    def reach(top, lev):
        """Distance from a level-`top` zone within which cells are at least level lev + 1."""
        return sum(N_BUFFER * h0 / 2 ** j for j in range(lev + 1, top + 1))

    def ell_sd(y, z, aa, bb):
        F = (y / aa) ** 2 + ((z - b) / bb) ** 2 - 1
        g = 2 * np.hypot(y / aa ** 2, (z - b) / bb ** 2)
        return F / np.maximum(g, 1e-9)

    def solid_sd(e, xa):
        inside = (e < 0) & (xa < 0)
        return np.where(inside, np.maximum(e, xa), np.hypot(np.maximum(e, 0), np.maximum(xa, 0)))

    def fields(p):
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(y, z - b)
        env = solid_sd(ell_sd(y, z, a, b), np.maximum(-x, x - L))
        lum = solid_sd(ell_sd(y, z, ai, bi), np.maximum(tt - x, x - (L - te)))
        tub = solid_sd(r - ro, np.maximum(L - x, x - x1))
        bor = solid_sd(r - ri, np.maximum(L - te - x, x - x1))
        d_surf = np.min(np.abs([env, lum, tub, bor]), 0)
        fluid = ((lum < 0) | (bor < 0) | ((env > 0) & (tub > 0) & (x < x_end)))
        return d_surf, fluid, lum, bor, x * sa + z * ca, r

    def d_features(p):
        d = np.full(len(p), np.inf)
        order = np.argsort(p[:, 0])
        xs = p[order, 0]
        w = FEATURE_BAND + 3.0
        for hh in rims:
            i0, i1 = np.searchsorted(xs, [hh["x"] - w, hh["x"] + w])
            if i1 <= i0:
                continue
            idx = order[i0:i1]
            q = p[idx]
            n = np.array(hh["n"])
            for c in (np.array(hh["c"]), np.array(hh["c"]) - t * n):
                v = q - c
                vn = v @ n
                rho = np.linalg.norm(v - np.outer(vn, n), axis=1)
                d[idx] = np.minimum(d[idx], np.hypot(vn, rho - hh["r"]))
        return d

    xs = dom["x0"] + (np.arange(dom["nx"]) + 0.5) * h0
    ys = dom["y0"] + (np.arange(dom["ny"]) + 0.5) * h0
    zs = dom["z0"] + (np.arange(dom["nz"]) + 0.5) * h0
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    p = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    top = max(lv.values())
    counts = []
    for lev in range(top + 1):
        h = h0 / 2 ** lev
        hd = 0.866 * h
        d_surf, fluid, lum, bor, hw, r = fields(p)
        need = np.zeros(len(p), int)

        def rule(dist, top_):
            if top_ <= lev:
                return
            need[dist - hd < reach(top_, lev)] = np.maximum(need[dist - hd < reach(top_, lev)], lev + 1)
        rule(d_surf, lv["skin"])
        rule(np.maximum(hw - h_pool, 0), lv["pool"])
        rule(np.maximum(lum, 0), lv["lumen"])
        rule(np.maximum(solid_sd(r - ri, np.maximum(L - te - p[:, 0], p[:, 0] - x1 - 1)), 0), lv["lumen"])
        rule(np.maximum(solid_sd(r - ro - 0.5, np.maximum(L - te - 2 - p[:, 0], p[:, 0] - L - 3)), 0), lv["bore"])
        for xa, xb in ((-0.6, tt + 0.6), (L - te - 0.6, L + 0.6)):
            boxd = np.maximum.reduce([xa - p[:, 0], p[:, 0] - xb, np.abs(p[:, 1]) - a - 0.6,
                                      -0.6 - p[:, 2], p[:, 2] - 2 * b - 0.6])
            rule(np.maximum(boxd, 0), lv["holes"])
        if lv["holes"] > lev:
            cand = d_surf < FEATURE_BAND + hd + reach(lv["holes"], lev) + 0.7
            df = np.full(len(p), np.inf)
            df[cand] = d_features(p[cand])
            rule(np.maximum(df - FEATURE_BAND, 0), lv["holes"])
        split = need > lev
        # a leaf counts if its centre is in the fluid; cells far inside the solid or beyond the pool
        # end wall are dropped before they are split further
        counts.append(int((~split & fluid).sum()))
        keep = split & (fluid | (d_surf < 2 * hd))
        c = p[keep]
        off = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * h / 4
        p = (c[:, None, :] + off[None]).reshape(-1, 3)
    return counts


def runtime_estimate(model, body, dom, info, res, cells, start, air_factor, cost_us=None, cores=48, eff=0.8):
    """Physical time, time steps and wall time of one run, from the model's quasi-static drainage
    curve through the CFD pool (free-surface area from the pool table)."""
    lv = RES[res]
    h_bore = GEO["base"] / 2 ** lv["bore"] * 1e-3
    h_hole = GEO["base"] / 2 ** lv["holes"] * 1e-3
    rho_g = model.gas.rho * air_factor
    rho_l = model.fluid.rho
    ps = model.suction.p_set
    r = info["curve"]
    hs, Vp = info["pool_h"], info["pool_V"]
    A = np.interp(r["h"], hs, free_surface_area(hs, Vp))          # mm2
    q = r["qL"] * 1e9                                                # mm3/s
    h0 = info["h0_" + start]
    qthr = Q_STOP_ML_MIN * 1e3 / 60
    sel = (r["h"] <= h0 + 1e-9)
    hh, qq, AA, pg, mode = r["h"][sel], q[sel], A[sel], r["pg"][sel], r["mode"][sel]
    run = qq > qthr
    n = np.argmin(run) if not run.all() else len(qq)
    dt_phys = np.zeros(n)
    dt_phys[1:] = 0.5 * (AA[1:n] / qq[1:n] + AA[:n - 1] / qq[:n - 1]) * np.diff(-hh[:n])
    md = max_delta_t(res, model.fluid, model.gas)
    gas = mode[:n] >= 2
    if start == "short":
        gas[:] = True
    u_l = np.sqrt(2 * np.maximum(np.abs(ps) + rho_l * 9.81 * 0.03, 0) / rho_l)
    u_bore = np.where(gas, np.sqrt(2 * np.maximum(pg[:n] - ps, 0) / rho_g), u_l)
    u_hole = np.where(gas, np.sqrt(2 * np.abs(np.minimum(pg[:n], 0)) / rho_g), u_l)
    dt = np.minimum.reduce([np.full(n, md), MAX_CO * h_bore / np.maximum(u_bore, 1e-9),
                            MAX_CO * h_hole / np.maximum(u_hole, 1e-9)])
    steps = float(np.sum(dt_phys / dt))
    t_phys = float(dt_phys.sum())
    tail = STOP_WINDOW_S + (0.3 if start == "short" else 0.0)       # stop window, start-up transient
    steps += tail / dt[-1]
    t_phys += tail
    s_step = cells * (cost_us or COST_US) * 1e-6 / (cores * eff)
    wall_h = steps * s_step / 3600
    t_gas = float(dt_phys[gas].sum()) + tail
    return dict(case=info["name"], start=start, res=res, air=air_factor, cells=int(cells), h0=h0,
                h_end=float(hh[n - 1]), t_phys=t_phys, t_gas=t_gas, dt_gas_us=float(dt[-1] * 1e6),
                steps=steps, s_step=s_step, wall_h=wall_h, core_h=wall_h * cores)


def draw_domain(body, dom, infos, rims, path):
    """Side view (world frame, to scale) and section of the VOF domain, for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    al = math.radians(TILT)
    sa, ca = math.sin(al), math.cos(al)
    a, b, L = body.a, body.b, body.L
    ri, ro = body.rp, body.rp + GEO["tube_wall"]

    def world(x, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        return x * ca - z * sa, x * sa + z * ca

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[3.2, 1]))
    A = ax[0]
    box = np.array([[dom["x0"], dom["z0"]], [dom["x_end"], dom["z0"]], [dom["x_end"], dom["z1"]],
                    [dom["x0"], dom["z1"]], [dom["x0"], dom["z0"]]])
    A.plot(*world(box[:, 0], box[:, 1]), color="0.3", lw=1.2, label="pool (trough) walls")
    tb = np.array([[dom["x_end"], b - ri], [dom["x1"], b - ri], [dom["x1"], b + ri], [dom["x_end"], b + ri]])
    A.plot(*world(tb[:, 0], tb[:, 1]), color="0.3", lw=1.2)
    dr = np.array([[0, 0], [L, 0], [L, b - ri], [L, b + ri], [L, 2 * b], [0, 2 * b], [0, 0]])
    A.fill(*world(dr[:, 0], dr[:, 1]), color="0.85", ec="k", lw=1, label="drain (side view)")
    A.plot(*world([L, L], [b - ri, b + ri]), color="w", lw=2.5)
    xo, zo = world(dom["x1"], b)
    A.annotate("outlet, −933 Pa", (xo, zo), (xo - 28, zo + 9), arrowprops=dict(arrowstyle="->"), fontsize=8)
    X0, X1 = world(dom["x0"], dom["z0"])[0] - 2, world(dom["x1"], dom["z1"])[0] + 2
    styles = {"usz_tilt10": "C3", "rec_tilt10": "C0"}
    for name, inf in infos.items():
        col = styles[name]
        A.axhline(inf["h_stop"], color=col, ls="--", lw=1, label=f"{name}: model stop {inf['h_stop']:.1f} mm")
        A.axhline(inf["h0_short"], color=col, ls=":", lw=1, label=f"{name}: short start {inf['h0_short']:.1f} mm")
    A.axhline(infos[next(iter(infos))]["h0_full"], color="k", ls="-.", lw=1,
              label=f"full start {infos[next(iter(infos))]['h0_full']:.1f} mm")
    lip = infos[next(iter(infos))]["lip"]
    xl, zl = world(L, b - ri)
    A.plot([xl], [zl], "o", color="C1", ms=4)
    A.annotate(f"bore lip {lip:.1f} mm", (xl, zl), (xl - 40, zl + 4), arrowprops=dict(arrowstyle="->"), fontsize=8)
    for ln in level_lines(body, dom, sa, ca)[::2]:
        p = ln["points"]
        A.plot(*world(p[:, 0], p[:, 2]), color="C2", lw=0.6, alpha=0.7)
    A.plot([], [], color="C2", lw=0.6, label="level probe lines")
    A.set(xlim=(X0, X1), aspect="equal", xlabel="horizontal (mm)", ylabel="height h above the lowest point of the skin (mm)",
          title=f"Side view, outlet end {TILT:g}° up (world frame, to scale)")
    A.legend(fontsize=7, loc="upper left", ncol=2)
    B = ax[1]
    B.add_patch(plt.Rectangle((dom["y0"], dom["z0"]), dom["y1"] - dom["y0"], dom["z1"] - dom["z0"], fill=False, ec="0.3"))
    th = np.linspace(-np.pi, np.pi, 400)
    B.fill(*body.outer(th), color="0.85", ec="k")
    B.fill(*body.inner(th), color="w", ec="k", lw=0.6)
    B.add_patch(plt.Circle((0, b), ri, fill=False, ls="--", ec="C1"))
    for h in rims:
        if h["x"] < 30:
            B.plot(h["outer"][:, 1], h["outer"][:, 2], color="C0", lw=0.5)
    B.set(aspect="equal", xlabel="y (mm)", ylabel="z, drain frame (mm)", title="Section (drain frame)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================ main
def prepare(name, cfg):
    pat = build(cfg["spec"])
    drain, fluid, gas, suction = Drain(), Fluid(), Gas(), Suction()
    model = DrainModel(pat, TILT, drain=drain, fluid=fluid, gas=gas, suction=suction)
    ana = model.analytic(dh=0.02)
    r = model.descend(dh=0.1)
    h_top = model.Z_top - model.Z_min
    info = dict(name=name, label=cfg["label"], spec=cfg["spec"], h_stop=r["h_stop"], lip=model.lip_height(),
                h_top=h_top, h_ana_simple=ana["h_stop_simple"], q_port=model.q_port_max(),
                h0_full=h_top + 3.0, h0_short=min(r["h_stop"] + SHORT_ABOVE, h_top + 3.0),
                curve=model.descend(dh=0.05, refine=False))
    return model, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "cases"))
    ap.add_argument("--only", nargs="*", default=list(CASES))
    ap.add_argument("--no-estimate", action="store_true")
    args = ap.parse_args()
    al = math.radians(TILT)
    sa, ca = math.sin(al), math.cos(al)
    body = Body(Drain())
    est, infos = [], {}
    for name in args.only:
        model, info = prepare(name, CASES[name])
        dom = domain(body, sa, ca, info["h0_full"])
        rims = hole_rims(body, model.pattern, GEO["rim_dx"])
        info["pool_h"], info["pool_V"] = pool_table(body, dom, sa, ca)
        # upper limit of the run: twice the model's time to stop (the watcher normally stops it first)
        for start in ("full", "short"):
            e = runtime_estimate(model, body, dom, info, "coarse", 1.0, start, 1.0)
            info[f"t_end_{start}"] = float(f"{2 * e['t_phys'] + 1:.2g}")
        write_case(name, info, body, rims, dom, model, model.fluid, model.gas, model.suction, sa, ca, args.out)
        print(f"  model: stop {info['h_stop']:.2f} mm, lip {info['lip']:.2f} mm, start full {info['h0_full']:.2f} "
              f"/ short {info['h0_short']:.2f} mm; domain {dom['nx']}x{dom['ny']}x{dom['nz']} background cells")
        infos[name] = info
        if name == "rec_tilt10":
            try:
                draw_domain(body, dom, infos, rims, os.path.join(HERE, "domain.png"))
            except ImportError:
                print("  matplotlib not available - domain.png not redrawn")
        if args.no_estimate:
            continue
        for res in RES:
            counts = octree_cells(body, rims, dom, res, info["h0_full"] + 1.0)
            print(f"  {res}: about {sum(counts):,} cells (per level {counts})")
            for start in ("full", "short"):
                for air in (1.0, 10.0):
                    est.append(runtime_estimate(model, body, dom, info, res, sum(counts), start, air))
    if est:
        write_estimate(est)


def write_estimate(est):
    rows = ["| Case | Start | Mesh | Air density | Cells | Physical time (s) | Two-phase part (s) | "
            "Δt two-phase (µs) | Time steps | s/step (48 cores) | Wall time (days) | Core-hours |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in est:
        rows.append(f"| {e['case']} | {e['start']} ({e['h0']:.1f} mm) | {e['res']} | ×{e['air']:g} | "
                    f"{e['cells'] / 1e6:.2f} M | {e['t_phys']:.1f} | {e['t_gas']:.1f} | {e['dt_gas_us']:.1f} | "
                    f"{e['steps'] / 1e6:.2f} M | {e['s_step']:.2f} | {e['wall_h'] / 24:.1f} | {e['core_h']:,.0f} |")
    txt = textwrap.dedent(f"""\
        # Run-time estimate (written by vof/make_cases.py)

        These are estimates. They are meant for planning; the test job (`vof/hpc/test_vof.sbatch`)
        measures the real cost per step on Euler, and `vof/analyse_run.py --timing <run>` rescales them.

        - **Cells:** octree estimate of the snappyHexMesh refinement (same levels, {N_BUFFER} buffer cells
          between levels). Expect ±30 %; `log.checkMesh` gives the real count.
        - **Physical time:** the model's quasi-static drainage curve Q_L(h) through the CFD pool,
          t = ∫ A_fs(h) / Q_L(h) dh, from the start level until Q_L < {Q_STOP_ML_MIN:g} mL/min, plus the
          {STOP_WINDOW_S:g} s stop window (and 0.3 s of start-up for the short start).
        - **Time step:** maxCo = maxAlphaCo = {MAX_CO:g}. Once air reaches the bore, it limits the step:
          the air jets through the bore at sqrt(2 (p_g - p_s) / ρ_air) ≈ 30–40 m/s, on bore cells of
          0.2 mm (coarse) or 0.1 mm (fine). Before that, the capillary limit (maxDeltaT) and the
          liquid velocity set it.
        - **Cost per step:** {COST_US:g} µs per cell per core (interFoam, PIMPLE with 3 pressure correctors,
          GAMG), 48 cores at 80 % parallel efficiency. This is the number to calibrate with the test job.
        - **Air ×10:** air density 12.04 kg/m³ instead of 1.204. Both resistances in series, the exposed
          holes and the bore, scale with ρ_air, so the vacuum in the drain and the stop level do not change.
          The air velocities drop by √10, and the time step rises by about 3. This is a numerical
          shortcut; use it for the long runs if real air is too expensive.

        """) + "\n".join(rows) + "\n"
    path = os.path.join(HERE, "runtime_estimate.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    json.dump(est, open(os.path.join(HERE, "runtime_estimate.json"), "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()

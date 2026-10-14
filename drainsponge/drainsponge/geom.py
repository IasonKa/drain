"""
Single-hole window of the USZ drain with a sponge-like (biphasic) tissue resting on it.

Two drain designs are supported (see config.DESIGNS):
  "current"  : sharp Ø1.0 hole in the 0.6 mm elliptic wall, tissue g0 above the skin
  "standoff" : same hole with a standoff ring around it (new_offset_desing.zip):
               ring r 0.55..0.85 mm, 0.40 mm high, cut by 4 parallel slots 0.35 mm wide
               (along and across the drain axis); the tissue rests g0 above the ring tops

Everything is built as ONE structured hexahedral mesh so that the fluid side
(OpenFOAM) and the tissue side (FEBio) share the interface nodes exactly:

        z (hole axis, outward)
        ^
        |   tissue  (biphasic sponge, H thick)       FEBio  / OpenFOAM porous zone
        |   ------------------------------------  <- tissue_iface (moves)
        |   film    (liquid above the ring tops)     OpenFOAM
        |   stand   ==ring==  slot  ==ring==          standoff (rigid), only for "standoff"
        |   ====skin====|  hole  |====skin====    <- drain_skin  (rigid)
        |   wall 0.6 mm |  (fluid)|  wall          hole_wall
        |   ====under===|        |====under===    <- wall_under
        |   channel (sheath channel inside drain)    outlet at y = +half (suction)
        +---------------------------------------> x (lateral)      y = drain axis

Local frame = the frame of drainfeb (extract_hole_patch.py / drain_geom.py):
origin on the outer skin at the hole axis, X lateral, Y along the drain, Z = hole axis.
The outer skin is the measured elliptic cylinder (8.925 x 4.835 mm semi-axes); the hole
sits at lateral -0.786 mm (case "hole_center"), so the skin is tilted ~2.7 deg.

Plan view: O-grid around the hole (centre square, inner ring up to the rim, outer ring
up to the square window edge). With a standoff, the outer ring has circular stations at
the ring radii and the angular node lines follow the parallel slot walls exactly.
Each plan node carries a column of nodes:
   channel | wall (fluid only in the hole) | stand (fluid except ring arcs) | film | tissue
The tissue bottom (= film top) is a height field over the fixed plan grid, so a deformed
tissue from FEBio is put back into the fluid mesh by moving nodes vertically only
(same topology -> OpenFOAM fields can be restarted).
Units: mm.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
import numpy as np

A_OUT, B_OUT = 8.925, 4.835          # outer skin semi-axes (mm), measured from drainage-3D.stp

ZONES = ("channel", "hole", "stand", "film", "tissue")
Z_CHAN, Z_HOLE, Z_STAND, Z_FILM, Z_TIS = range(5)
# plan quad regions
R_CENTRE, R_INNER, R_OUTER, R_RING, R_SLOT = range(5)
# hex faces (OpenFOAM / VTK hex numbering), outward for a right-handed hex
HEX_FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
FACE_ZMIN, FACE_ZMAX = 4, 5


@dataclass
class Window:
    # geometry (mm) - defaults = current design (drainage-3D, sharp 1.0 mm holes)
    hole_d: float = 1.0
    wall: float = 0.6
    fillet: float = 0.0             # rounded outer hole edge (0 = sharp, as in the CAD)
    half: float = 1.4               # half width of the square window (stays clear of neighbour holes/rings)
    gap: float = 0.05               # initial liquid film between tissue and the highest drain feature
    tissue_h: float = 2.0           # tissue thickness modelled above the window
    channel_d: float = 1.0          # depth of the sheath channel under the wall
    lateral: float = -0.7864068576737462   # hole_center.json
    # standoff ring (0 height = no standoff = current design)
    standoff_h: float = 0.0
    standoff_rin: float = 0.55
    standoff_rout: float = 0.85
    slot_w: float = 0.35
    n_slots: int = 4
    slot_angle0: float = 0.0        # deg, direction of the first slot (0 = lateral X; 90 = drain axis)
    # resolution
    nq: int = 16                    # cells per quarter around the hole
    n_in: int = 6                   # radial cells: centre square -> rim
    n_gap: int = 2                  # radial cells: rim -> ring inner radius
    n_ring: int = 6                 # radial cells across the ring
    n_out: int = 20                 # radial cells: rim (or ring) -> window edge
    n_chan: int = 8
    n_wall: int = 10
    n_stand: int = 8                # vertical cells skin -> ring top
    n_film: int = 5
    n_tis: int = 24
    rim_cell: float = 0.015         # first radial cell outside the rim / ring (mm)
    tis_cell: float = 0.02          # first tissue layer at the interface (mm)
    h_min: float = 0.003            # residual film where tissue touches the drain (surface roughness)

    @property
    def a(self):
        return 0.5 * self.hole_d

    @property
    def has_standoff(self):
        return self.standoff_h > 0

    @property
    def top_feature(self):          # height of the highest drain feature above the skin
        return self.standoff_h if self.has_standoff else 0.0

    @property
    def xc(self):                   # ellipse centre (local X)
        return -self.lateral

    @property
    def zc(self):                   # ellipse centre (local Z)
        return -B_OUT * np.sqrt(1.0 - (self.lateral / A_OUT) ** 2)

    def skin_e(self, x):
        """Outer elliptic skin height at local x (no hole edge treatment)."""
        lat = np.asarray(x, float) - self.xc
        return B_OUT * np.sqrt(np.clip(1.0 - (lat / A_OUT) ** 2, 0.0, None)) + self.zc

    def skin(self, x, y):
        """Skin height including the optional fillet at the hole edge."""
        z = self.skin_e(x)
        f = self.fillet
        if f <= 0:
            return z
        r = np.hypot(x, y)
        d = np.clip(self.a + f - r, 0.0, f)
        return z - (f - np.sqrt(np.clip(f * f - d * d, 0.0, None)))

    def slot_dirs(self):
        ang = np.radians(self.slot_angle0) + 2 * np.pi * np.arange(self.n_slots) / max(self.n_slots, 1)
        return np.c_[np.cos(ang), np.sin(ang)]

    def in_slot(self, xy):
        """True where a plan point lies inside one of the slots (parallel walls, width slot_w)."""
        xy = np.atleast_2d(xy)
        out = np.zeros(len(xy), bool)
        for d in self.slot_dirs():
            n = np.array([-d[1], d[0]])
            out |= (xy @ d > 0) & (np.abs(xy @ n) < 0.5 * self.slot_w)
        return out

    def to_json(self, path):
        json.dump(asdict(self), open(path, "w"), indent=1)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# ---------------------------------------------------------------------- 1D spacing
def graded(n, first):
    """n+1 points in [0, 1], geometric growth, first interval ~= first (fraction)."""
    if first * n >= 1.0:
        return np.linspace(0.0, 1.0, n + 1)
    lo, hi = 1.0, 2.0
    while first * (hi ** n - 1) / (hi - 1) < 1.0:
        hi *= 2
    for _ in range(200):                                   # solve first*(q^n-1)/(q-1) = 1
        q = 0.5 * (lo + hi)
        if first * (q ** n - 1) / (q - 1) < 1.0:
            lo = q
        else:
            hi = q
    q = 0.5 * (lo + hi)
    s = np.concatenate([[0.0], np.cumsum(first * q ** np.arange(n))])
    return s / s[-1]


# ---------------------------------------------------------------------- angular distribution
class Angular:
    """Angles of the 4*nq angular node lines as a function of radius.

    Anchors: the 4 quarter corners (45 + 90 m deg, they run into the window corners)
    and, with a standoff, both walls of every slot (parallel lines -> angle depends on r).
    Node counts between anchors are fixed (structured grid); anchor angles move with r.
    """

    def __init__(self, w: Window):
        self.w = w
        nq = w.nq
        self.N = 4 * nq
        self.corner = np.radians(45.0 + 90.0 * np.arange(4))
        self.walls = []                                    # (direction, offset sign)
        if w.has_standoff:
            for d in w.slot_dirs():
                for sg in (-1, 1):
                    self.walls.append((d, sg))
        r_ref = 0.5 * (w.standoff_rin + w.standoff_rout) if w.has_standoff else w.a
        ang, kind = self._anchors(r_ref)
        # cells per anchor segment: every quarter (corner to corner) gets exactly nq cells,
        # split over its segments in proportion to their angle (largest remainder, >= 1 each)
        seg = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        cnt = np.zeros(len(seg), int)
        corners = [i for i, k in enumerate(kind) if k == "corner"] + [len(kind)]
        for q in range(4):
            ids = np.arange(corners[q], corners[q + 1])
            share = seg[ids] / seg[ids].sum() * nq
            c = np.maximum(1, np.floor(share).astype(int))
            while c.sum() > nq:
                c[np.argmax(c - share)] -= 1
            while c.sum() < nq:
                c[np.argmax(share - c)] += 1
            if np.any(c < 1):
                raise RuntimeError("nq too small for the slots")
            cnt[ids] = c
        self.cnt = cnt
        self.kind0 = kind

    def _anchors(self, r, edge=False):
        """Sorted anchor angles at radius r (or on the window edge), starting at the first corner."""
        L = self.w.half
        a = [(c, "corner") for c in self.corner]
        for d, sg in self.walls:
            n = np.array([-d[1], d[0]])
            off = sg * 0.5 * self.w.slot_w
            if edge:
                # intersection of the wall line s*d + off*n with the square boundary
                best = None
                for s in np.linspace(0, 3 * L, 30001):
                    p = s * d + off * n
                    if max(abs(p[0]), abs(p[1])) >= L:
                        best = p
                        break
                th = np.arctan2(best[1], best[0])
            else:
                rr = max(r, abs(off) + 1e-9)
                p = np.sqrt(rr * rr - off * off) * d + off * n
                th = np.arctan2(p[1], p[0])
            a.append((th, "wall"))
        base = self.corner[0]
        a = [((t - base) % (2 * np.pi) + base, k) for t, k in a]
        a.sort()
        ang = np.array([t for t, _ in a])
        kind = [k for _, k in a]
        return ang, kind

    def angles(self, r=None, edge=False):
        ang, kind = self._anchors(r if r is not None else self.w.a, edge)
        if kind != self.kind0:
            raise RuntimeError("slot walls cross a window corner direction - change slot_angle0")
        out = []
        ang2 = np.r_[ang, ang[0] + 2 * np.pi]
        for i, c in enumerate(self.cnt):
            out.append(ang2[i] + (ang2[i + 1] - ang2[i]) * np.arange(c) / c)
        out = np.concatenate(out)
        return out                                          # length N, starts at the first corner


# ---------------------------------------------------------------------- plan O-grid
def plan_grid(w: Window):
    """Plan nodes (N,2), plan quads (M,4) CCW, region of each quad, radial station radii."""
    a, L, nq = w.a, w.half, w.nq
    c = 0.42 * a                                           # centre square half-size
    ag = Angular(w)
    # every quarter must contain nq angular cells so the centre square closes
    th_a = ag.angles(a)
    corner_idx = np.nonzero(np.isin(np.round(th_a, 12), np.round(ag.corner, 12)))[0]
    if len(corner_idx) != 4 or not np.all(np.diff(np.r_[corner_idx, corner_idx[0] + ag.N]) == nq):
        raise RuntimeError("angular anchors do not leave nq cells per quarter; adjust nq / slots")

    def ring_pts(r, edge=False):
        th = ag.angles(r, edge)
        th = np.r_[th, th[0] + 2 * np.pi]                  # closed (N+1)
        if edge:
            # point on the square boundary at angle th
            cth, sth = np.cos(th), np.sin(th)
            s = L / np.maximum(np.abs(cth), np.abs(sth))
            return np.c_[s * cth, s * sth]
        return np.c_[r * np.cos(th), r * np.sin(th)]

    def square_pts(h):
        """Centre-square boundary points matching the ring angles at r = a (closed, N+1)."""
        th = ag.angles(a)
        th = np.r_[th, th[0] + 2 * np.pi]
        cth, sth = np.cos(th), np.sin(th)
        s = h / np.maximum(np.abs(cth), np.abs(sth))
        return np.c_[s * cth, s * sth]

    # radial stations of the annular block (all angular lines, closed loop)
    stations = [("square", square_pts(c))]
    for t in np.linspace(0, 1, w.n_in + 1)[1:]:
        stations.append(("blend", (square_pts(c), ring_pts(a), t)))
    if w.has_standoff:
        for r in np.linspace(a, w.standoff_rin, w.n_gap + 1)[1:]:
            stations.append(("circle", r))
        for r in np.linspace(w.standoff_rin, w.standoff_rout, w.n_ring + 1)[1:]:
            stations.append(("circle", r))
        r0 = w.standoff_rout
    else:
        r0 = a
    t_out = graded(w.n_out, w.rim_cell / (L - r0))[1:]
    P_r0, P_edge = ring_pts(r0), ring_pts(r0, edge=True)
    for t in t_out:
        stations.append(("blend", (P_r0, P_edge, t)))
    rings = []
    for kind, v in stations:
        if kind == "square":
            rings.append(v)
        elif kind == "circle":
            rings.append(ring_pts(v))
        else:
            p0, p1, t = v
            rings.append(p0 + t * (p1 - p0))
    rings = np.stack(rings)                                # (S, N+1, 2)
    S = rings.shape[0]
    n_in = w.n_in
    # centre square: tensor grid in each quarter's local frame is not straightforward with
    # non-uniform angles, so build it by transfinite interpolation of its 4 edges
    sq = rings[0][:-1]                  # N centre-square boundary points, index 0 = corner (c, c) at 45 deg
    bottom = sq[2 * nq:3 * nq + 1]                         # (-c,-c) -> ( c,-c)
    right = np.r_[sq[3 * nq:4 * nq], sq[:1]]               # ( c,-c) -> ( c, c)
    top = sq[0:nq + 1][::-1]                               # (-c, c) -> ( c, c)
    left = sq[nq:2 * nq + 1][::-1]                         # (-c,-c) -> (-c, c)
    u = np.linspace(0, 1, nq + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    C = ((1 - V)[..., None] * bottom[:, None, :] + V[..., None] * top[:, None, :]
         + (1 - U)[..., None] * left[None, :, :] + U[..., None] * right[None, :, :]
         - ((1 - U) * (1 - V))[..., None] * bottom[0] - (U * (1 - V))[..., None] * bottom[-1]
         - ((1 - U) * V)[..., None] * top[0] - (U * V)[..., None] * top[-1])
    pts = [C.reshape(-1, 2), rings.reshape(-1, 2)]
    nC = (nq + 1) ** 2
    idxC = np.arange(nC).reshape(nq + 1, nq + 1)
    quads = [np.stack([idxC[:-1, :-1], idxC[1:, :-1], idxC[1:, 1:], idxC[:-1, 1:]], -1).reshape(-1, 4)]
    reg = [np.full(nq * nq, R_CENTRE)]
    idxR = nC + np.arange(S * (ag.N + 1)).reshape(S, ag.N + 1)
    q4 = np.stack([idxR[:-1, :-1], idxR[1:, :-1], idxR[1:, 1:], idxR[:-1, 1:]], -1)   # (S-1, N, 4)
    quads.append(q4.reshape(-1, 4))
    st_reg = np.full(S - 1, R_OUTER)
    st_reg[:n_in] = R_INNER
    if w.has_standoff:
        st_reg[n_in + w.n_gap:n_in + w.n_gap + w.n_ring] = R_RING
    reg.append(np.repeat(st_reg, ag.N))
    pts = np.concatenate(pts)
    quads = np.concatenate(quads)
    reg = np.concatenate(reg)
    # merge coincident nodes
    key = np.round(pts / 1e-7).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    pts = pts[first]
    quads = inv[quads]
    # CCW orientation
    p = pts[quads]
    area = 0.5 * ((p[:, 2, 0] - p[:, 0, 0]) * (p[:, 3, 1] - p[:, 1, 1])
                  - (p[:, 3, 0] - p[:, 1, 0]) * (p[:, 2, 1] - p[:, 0, 1]))
    if np.any(np.abs(area) < 1e-14):
        raise RuntimeError("degenerate plan cell")
    quads[area < 0] = quads[area < 0][:, ::-1]
    # ring band: slots are fluid
    if w.has_standoff:
        ring = reg == R_RING
        cen = pts[quads[ring]].mean(1)
        sl = w.in_slot(cen)
        rr = np.nonzero(ring)[0]
        reg[rr[sl]] = R_SLOT
    return pts, quads, reg


# ---------------------------------------------------------------------- 3D mesh
class WindowMesh:
    """Structured hex mesh of channel + hole + stand + film + tissue. Build with WindowMesh(w)."""

    def __init__(self, w: Window, tissue_bottom=None, zones=ZONES):
        self.w = w
        self.zones = tuple(zones)
        self.plan, self.quads, self.region = plan_grid(w)
        x, y = self.plan.T
        r = np.hypot(x, y)
        a = w.a
        self.r = r
        self.node_hole = r < a - 1e-6              # strictly inside the hole
        self.node_rim = np.abs(r - a) < 1e-6
        self.cell_hole = np.isin(self.region, (R_CENTRE, R_INNER))
        self.cell_ring = self.region == R_RING
        self.node_ring = np.zeros(len(x), bool)
        self.node_ring[np.unique(self.quads[self.cell_ring])] = True
        n_stand = w.n_stand if w.has_standoff else 0
        nk = [w.n_chan, w.n_wall, n_stand, w.n_film, w.n_tis]
        self.nk = nk
        self.k0 = np.cumsum([0] + nk)              # first level of each layer
        self.NK = self.k0[-1]
        self.skin0 = w.skin(x, y)
        self.zrim = w.skin_e(x) - w.fillet
        self.ztop_feature = w.skin_e(x) + w.top_feature
        if tissue_bottom is None:
            tissue_bottom = w.skin_e(x) + w.top_feature + w.gap
        self.z_top = w.skin_e(x) + w.top_feature + w.gap + w.tissue_h
        self._build_cells()
        self.set_tissue_bottom(tissue_bottom)

    # ------------------------------------------------------------ levels
    def set_tissue_bottom(self, zt):
        """Place all column nodes for a given tissue-bottom height field (one value per plan node).

        Fixed levels (walls) stay; the levels that are only mesh surfaces (skin level inside the
        hole, ring-top level away from the ring) are placed proportionally between the fixed
        levels and the tissue bottom, so every layer keeps a positive thickness."""
        w = self.w
        x, y = self.plan.T
        zt = np.array(zt, float)
        ze = w.skin_e(x)
        hs = w.top_feature
        z1 = ze - w.wall                                   # wall bottom (fixed)
        z0 = z1 - w.channel_d                              # channel bottom (fixed)
        hn = self.node_hole
        ring = self.node_ring
        # highest fixed drain surface under each node
        floor = np.where(ring, ze + hs, self.skin0)
        floor[hn] = z1[hn] + 0.1 * w.wall
        self.intrusion = float(np.max(self.zrim[hn] - zt[hn])) if hn.any() else 0.0
        self.occluded = self.intrusion > 0.85 * (w.wall - w.fillet)
        zt = np.maximum(zt, floor + w.h_min)
        self.zt = zt
        # level 2 (skin / rim), level 3 (ring top)
        z2 = self.skin0.copy()
        z3 = ze + hs
        tot0 = w.wall + hs + w.gap                          # initial hole-column height above wall bottom
        f2 = (w.wall - w.fillet) / tot0
        f3 = (w.wall + hs) / tot0
        z2[hn] = z1[hn] + f2 * (zt[hn] - z1[hn])
        if hs > 0:
            free = ~ring
            g3 = hs / (hs + w.gap)
            base = np.where(hn, z1 + f2 * (zt - z1), z2)
            z3 = np.where(free, base + g3 * (zt - base), z3)
            z3[hn] = z1[hn] + f3 * (zt[hn] - z1[hn])
        else:
            z3 = z2
        K = self.NK + 1
        Z = np.empty((len(x), K))
        k0 = self.k0
        lin = lambda lo, hi, n: lo[:, None] + np.linspace(0, 1, n + 1)[None] * (hi - lo)[:, None]
        Z[:, k0[0]:k0[1] + 1] = lin(z0, z1, self.nk[0])
        Z[:, k0[1]:k0[2] + 1] = lin(z1, z2, self.nk[1])
        Z[:, k0[2]:k0[3] + 1] = lin(z2, z3, self.nk[2])
        Z[:, k0[3]:k0[4] + 1] = lin(z3, zt, self.nk[3])
        g = graded(w.n_tis, w.tis_cell / w.tissue_h)
        Z[:, k0[4]:k0[5] + 1] = zt[:, None] + g[None] * (self.z_top - zt)[:, None]
        self.Z = Z
        P = np.empty((len(x) * K, 3))
        P[:, 0] = np.repeat(x, K)
        P[:, 1] = np.repeat(y, K)
        P[:, 2] = Z.ravel()
        self.points_all = P
        self.points = P[self.used]

    def film_thickness(self):
        """Liquid layer above the highest drain surface under each plan node (hole: above the rim level)."""
        w = self.w
        ze = w.skin_e(self.plan[:, 0])
        base = np.where(self.node_ring, ze + w.top_feature, self.skin0)
        base[self.node_hole] = self.zrim[self.node_hole]
        return self.zt - base

    # ------------------------------------------------------------ topology
    def _build_cells(self):
        K = self.NK + 1
        cells, zone, layer, pcell = [], [], [], []
        for zi, (name, nl) in enumerate(zip(ZONES, self.nk)):
            if name not in self.zones or nl == 0:
                continue
            if name == "hole":
                sel = np.nonzero(self.cell_hole)[0]
            elif name == "stand":
                sel = np.nonzero(~self.cell_ring)[0]
            else:
                sel = np.arange(len(self.quads))
            q = self.quads[sel]
            for kk in range(nl):
                k = self.k0[zi] + kk
                lo = q * K + k
                cells.append(np.concatenate([lo, lo + 1], 1))
                zone.append(np.full(len(q), zi))
                layer.append(np.full(len(q), k))
                pcell.append(sel)
        cells = np.concatenate(cells)
        used = np.unique(cells)
        remap = -np.ones(len(self.plan) * K, np.int64)
        remap[used] = np.arange(len(used))
        self.used = used
        self.node_of_all = remap
        self.cells = remap[cells]
        self.cell_zone = np.concatenate(zone)
        self.cell_layer = np.concatenate(layer)
        self.cell_plan = np.concatenate(pcell)
        self.node_plan = used // K
        self.node_level = used % K
        self._faces()

    def _faces(self):
        """Unique faces, owner/neighbour, boundary classification."""
        C = self.cells
        nC = len(C)
        allf = np.concatenate([C[:, list(f)] for f in HEX_FACES])          # (6 nC, 4)
        own = np.tile(np.arange(nC), 6)
        loc = np.repeat(np.arange(6), nC)
        key = np.sort(allf, 1)
        _, first, inv, cnt = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        nf = len(first)
        order = np.argsort(inv, kind="stable")
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        i0 = order[starts]
        two = cnt == 2
        i1 = np.full(nf, -1)
        i1[two] = order[starts[two] + 1]
        if np.any(cnt > 2):
            raise RuntimeError("non-manifold face in the hex mesh")
        c0, c1 = own[i0], np.where(two, own[np.maximum(i1, 0)], -1)
        swap = two & (c1 < c0)
        inst_owner = np.where(swap, i1, i0)
        self.f_nodes = allf[inst_owner]
        self.f_owner = np.where(swap, c1, c0)
        self.f_neigh = np.where(swap, c0, c1)
        self.f_local = loc[inst_owner]
        self.f_internal = two
        self.f_patch = np.full(nf, "", dtype=object)
        self._classify()

    def _classify(self):
        w = self.w
        L = w.half
        b = ~self.f_internal
        own = self.f_owner
        zi = self.cell_zone[own]
        loc = self.f_local
        pl = self.plan[self.node_plan[self.f_nodes]]                     # (nf,4,2)
        on_edge = np.all((np.abs(np.abs(pl[..., 0]) - L) < 1e-6) | (np.abs(np.abs(pl[..., 1]) - L) < 1e-6), 1)
        lateral = loc < 4
        ring_cell = self.cell_ring[self.cell_plan[own]]
        P = self.f_patch
        y_plus = np.all(np.abs(pl[..., 1] - L) < 1e-6, 1)
        m = b & (zi == Z_CHAN)
        P[m & (loc == FACE_ZMIN)] = "channel_walls"
        P[m & (loc == FACE_ZMAX)] = "wall_under"
        P[m & lateral & on_edge & y_plus] = "outlet"
        P[m & lateral & on_edge & ~y_plus] = "channel_walls"
        m = b & (zi == Z_HOLE)
        P[m & lateral] = "hole_wall"
        m = b & (zi == Z_STAND)
        P[m & (loc == FACE_ZMIN)] = "drain_skin"
        P[m & lateral & on_edge] = "film_edge"
        P[m & lateral & ~on_edge] = "standoff"
        m = b & (zi == Z_FILM)
        P[m & (loc == FACE_ZMIN)] = np.where(ring_cell[m & (loc == FACE_ZMIN)], "standoff", "drain_skin")
        P[m & (loc == FACE_ZMAX)] = "tissue_iface"
        P[m & lateral] = "film_edge"
        m = b & (zi == Z_TIS)
        P[m & (loc == FACE_ZMAX)] = "tissue_top"
        P[m & lateral] = "tissue_side"
        P[m & (loc == FACE_ZMIN)] = "tissue_bottom"
        zn = self.cell_zone[np.maximum(self.f_neigh, 0)]
        self.f_iface = self.f_internal & (np.minimum(zi, zn) == Z_FILM) & (np.maximum(zi, zn) == Z_TIS)
        if np.any(b & (P == "")):
            raise RuntimeError(f"{np.sum(b & (P == ''))} unclassified boundary faces")

    # ------------------------------------------------------------ helpers
    def tissue_nodes(self):
        return np.nonzero(self.node_level >= self.k0[4])[0]

    def iface_node_ids(self):
        """Used-node id of the tissue-bottom node of every plan node (plan order)."""
        K = self.NK + 1
        return self.node_of_all[np.arange(len(self.plan)) * K + self.k0[4]]

    def summary(self):
        z = self.cell_zone
        h = self.film_thickness()
        return dict(points=int(len(self.points)), cells=int(len(self.cells)),
                    **{f"cells_{n}": int(np.sum(z == i)) for i, n in enumerate(ZONES)},
                    plan_nodes=int(len(self.plan)), plan_quads=int(len(self.quads)),
                    ring_quads=int(self.cell_ring.sum()), slot_quads=int(np.sum(self.region == R_SLOT)),
                    min_film_mm=float(h[~self.node_hole].min()),
                    intrusion_mm=self.intrusion, occluded=self.occluded)

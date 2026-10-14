"""
Single-hole window of the USZ drain with a sponge-like (biphasic) tissue resting on it.

Everything is built as ONE structured hexahedral mesh so that the fluid side
(OpenFOAM) and the tissue side (FEBio) share the interface nodes exactly:

        z (hole axis, outward)
        ^
        |   tissue  (biphasic sponge, H thick)       FEBio  / OpenFOAM porous zone
        |   ------------------------------------  <- tissue_iface (moves)
        |   film    (liquid gap, g0 = 50 um)         OpenFOAM
        |   ====skin====|  hole  |====skin====    <- drain_skin  (rigid)
        |   wall 0.6 mm |  (fluid)|  wall          hole_wall
        |   ====under===|        |====under===    <- wall_under
        |   channel (sheath channel inside drain)    outlet at y = +half (suction)
        +---------------------------------------> x (lateral)      y = drain axis

Local frame = the frame of drainfeb (extract_hole_patch.py / drain_geom.py):
origin on the outer skin at the hole axis, X lateral, Y along the drain, Z = hole axis.
The outer skin is the measured elliptic cylinder (8.925 x 4.835 mm semi-axes); the hole
sits at lateral -0.786 mm (case "hole_center", same as the FEBio film study), so the
skin is slightly tilted (~2.7 deg) relative to the hole axis, as in the real part.

Plan view: O-grid around the hole (centre square, inner ring up to the rim, outer ring
up to the square window edge). Each plan node carries a column of nodes:
    channel (n_chan) | wall segment (n_wall, fluid only inside the hole) | film (n_film) | tissue (n_tis)
The tissue bottom (= film top) is a height field over the fixed plan grid, so a deformed
tissue from FEBio is put back into the fluid mesh by moving film/hole nodes vertically
only (no remeshing, same topology -> OpenFOAM fields can be restarted).
Units: mm.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields
import numpy as np

A_OUT, B_OUT = 8.925, 4.835          # outer skin semi-axes (mm), measured from drainage-3D.stp

ZONES = ("channel", "hole", "film", "tissue")
# hex faces (OpenFOAM / VTK hex numbering), outward for a right-handed hex
HEX_FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))
FACE_ZMIN, FACE_ZMAX = 4, 5


@dataclass
class Window:
    # geometry (mm) - defaults = current design (drainage-3D, sharp 1.0 mm holes)
    hole_d: float = 1.0
    wall: float = 0.6
    fillet: float = 0.0             # rounded outer hole edge (0 = sharp, as in the CAD)
    half: float = 1.6               # half width of the square window (neighbour holes start at 1.75)
    gap: float = 0.05               # initial liquid film between tissue and drain ("almost touching")
    tissue_h: float = 2.0           # tissue thickness modelled above the window
    channel_d: float = 1.0          # depth of the sheath channel under the wall
    lateral: float = -0.7864068576737462   # hole_center.json
    # resolution
    nq: int = 16                    # cells per quarter around the hole
    n_in: int = 6                   # radial cells: centre square -> rim
    n_out: int = 20                 # radial cells: rim -> window edge
    n_chan: int = 8
    n_wall: int = 10
    n_film: int = 5
    n_tis: int = 24
    rim_cell: float = 0.015         # first radial cell outside the rim (mm)
    tis_cell: float = 0.02          # first tissue layer at the interface (mm)
    h_min: float = 0.003            # residual film where tissue touches the drain (surface roughness)

    @property
    def a(self):
        return 0.5 * self.hole_d

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


# ---------------------------------------------------------------------- plan O-grid
def plan_grid(w: Window):
    """Plan nodes (N,2), plan quads (M,4) CCW, region of each quad (0 centre, 1 inner ring, 2 outer ring)."""
    a, L, nq = w.a, w.half, w.nq
    c = 0.42 * a                                           # centre square half-size (corners at 0.6 a)
    s = np.linspace(-1.0, 1.0, nq + 1)
    th = s * np.pi / 4
    blocks = []                                            # (P[i,j,2], region)
    # centre square (tan spacing so its edges match the ring rays)
    u = c * np.tan(th)
    X, Y = np.meshgrid(u, u, indexing="ij")
    blocks.append((np.stack([X, Y], -1), 0))
    t_in = np.linspace(0.0, 1.0, w.n_in + 1)
    t_out = graded(w.n_out, w.rim_cell / (L - a))
    for q in range(4):
        R = np.array([[np.cos(q * np.pi / 2), -np.sin(q * np.pi / 2)],
                      [np.sin(q * np.pi / 2), np.cos(q * np.pi / 2)]])
        Pc = np.stack([np.full_like(th, c), c * np.tan(th)], -1)       # centre-square side
        Pa = np.stack([a * np.cos(th), a * np.sin(th)], -1)            # rim
        PL = np.stack([np.full_like(th, L), L * np.tan(th)], -1)       # window edge
        inner = Pc[None] + t_in[:, None, None] * (Pa - Pc)[None]
        outer = Pa[None] + t_out[:, None, None] * (PL - Pa)[None]
        blocks.append((inner @ R.T, 1))
        blocks.append((outer @ R.T, 2))
    pts, quads, reg = [], [], []
    off = 0
    for P, rg in blocks:
        ni, nj = P.shape[:2]
        pts.append(P.reshape(-1, 2))
        idx = np.arange(ni * nj).reshape(ni, nj) + off
        q4 = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        quads.append(q4)
        reg.append(np.full(len(q4), rg))
        off += ni * nj
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
    quads[area < 0] = quads[area < 0][:, ::-1]
    return pts, quads, reg


# ---------------------------------------------------------------------- 3D mesh
class WindowMesh:
    """Structured hex mesh of channel + hole + film + tissue. Build with WindowMesh(w)."""

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
        self.cell_hole = self.region < 2           # plan quads inside the hole
        nk = [w.n_chan, w.n_wall, w.n_film, w.n_tis]
        self.k0 = np.cumsum([0] + nk)              # first level of each layer
        self.NK = self.k0[-1]
        self.skin0 = w.skin(x, y)
        self.zrim = w.skin_e(x) - w.fillet
        if tissue_bottom is None:
            tissue_bottom = w.skin_e(x) + w.gap
        self.z_top = w.skin_e(x) + w.gap + w.tissue_h          # tissue top (held by the organ)
        self._build_cells()
        self.set_tissue_bottom(tissue_bottom)

    # ------------------------------------------------------------ levels
    def set_tissue_bottom(self, zt):
        """Place all column nodes for a given tissue-bottom height field (one value per plan node)."""
        w = self.w
        x, y = self.plan.T
        zt = np.array(zt, float)
        out = ~self.node_hole
        # the film may not be thinner than h_min above the skin / rim
        zt[out] = np.maximum(zt[out], self.skin0[out] + w.h_min)
        ze = w.skin_e(x)
        z_cb = ze - w.wall - w.channel_d
        z_wb = ze - w.wall
        z_s = self.skin0.copy()
        # inside the hole the skin level is only an internal mesh surface: keep a film layer
        # at least as thick as the film at the rim, and keep the wall segment open
        h_rim = max(float(np.mean(zt[self.node_rim] - self.zrim[self.node_rim])), w.h_min)
        hn = self.node_hole
        z_s[hn] = np.minimum(self.zrim[hn], zt[hn] - h_rim)
        lim = z_wb[hn] + 0.1 * w.wall
        self.intrusion = float(np.max(self.zrim[hn] - zt[hn]))       # depth of tissue below the rim
        self.occluded = self.intrusion > 0.85 * (w.wall - w.fillet)  # tissue reaches the channel
        z_s[hn] = np.maximum(z_s[hn], lim)
        zt[hn] = np.maximum(zt[hn], z_s[hn] + w.h_min)
        self.zt = zt
        Z = np.empty((len(x), self.NK + 1))
        k0 = self.k0
        Z[:, k0[0]:k0[1] + 1] = z_cb[:, None] + np.linspace(0, 1, w.n_chan + 1)[None] * (z_wb - z_cb)[:, None]
        Z[:, k0[1]:k0[2] + 1] = z_wb[:, None] + np.linspace(0, 1, w.n_wall + 1)[None] * (z_s - z_wb)[:, None]
        Z[:, k0[2]:k0[3] + 1] = z_s[:, None] + np.linspace(0, 1, w.n_film + 1)[None] * (zt - z_s)[:, None]
        g = graded(w.n_tis, w.tis_cell / w.tissue_h)
        Z[:, k0[3]:k0[4] + 1] = zt[:, None] + g[None] * (self.z_top - zt)[:, None]
        self.Z = Z
        P = np.empty((len(x) * (self.NK + 1), 3))
        P[:, 0] = np.repeat(x, self.NK + 1)
        P[:, 1] = np.repeat(y, self.NK + 1)
        P[:, 2] = Z.ravel()
        self.points_all = P
        self.points = P[self.used]

    def film_thickness(self):
        """Film thickness above each plan node (tissue bottom - skin/rim level)."""
        h = self.zt - self.skin0
        h[self.node_hole] = self.zt[self.node_hole] - self.zrim[self.node_hole]
        return h

    # ------------------------------------------------------------ topology
    def _build_cells(self):
        w = self.w
        K = self.NK + 1
        cells, zone, layer, pcell = [], [], [], []
        for zi, (name, nl) in enumerate(zip(ZONES, (w.n_chan, w.n_wall, w.n_film, w.n_tis))):
            if name not in self.zones:
                continue
            sel = np.arange(len(self.quads)) if name != "hole" else np.nonzero(self.cell_hole)[0]
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
        # plan-node / level of every used node
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
        # for every unique face: the two (or one) cell-face instances
        order = np.argsort(inv, kind="stable")
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        i0 = order[starts]
        two = cnt == 2
        i1 = np.full(nf, -1)
        i1[two] = order[starts[two] + 1]
        if np.any(cnt > 2):
            raise RuntimeError("non-manifold face in the hex mesh")
        c0, c1 = own[i0], np.where(two, own[np.maximum(i1, 0)], -1)
        # internal: owner = lower cell id, face ordered as owner's outward face
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
        zone = np.array(ZONES)[self.cell_zone[own]]
        layer = self.cell_layer[own]
        loc = self.f_local
        k0 = self.k0
        pl = self.plan[self.node_plan[self.f_nodes]]                     # (nf,4,2)
        on_edge = np.all((np.abs(np.abs(pl[..., 0]) - L) < 1e-6) | (np.abs(np.abs(pl[..., 1]) - L) < 1e-6), 1)
        lateral = loc < 4
        P = self.f_patch
        y_plus = np.all(np.abs(pl[..., 1] - L) < 1e-6, 1)
        m = b & (zone == "channel")
        P[m & (loc == FACE_ZMIN)] = "channel_walls"
        P[m & (loc == FACE_ZMAX)] = "wall_under"
        P[m & lateral & on_edge & y_plus] = "outlet"
        P[m & lateral & on_edge & ~y_plus] = "channel_walls"
        m = b & (zone == "hole")
        P[m & lateral] = "hole_wall"
        m = b & (zone == "film")
        P[m & (loc == FACE_ZMIN)] = "drain_skin"
        P[m & (loc == FACE_ZMAX)] = "tissue_iface"
        P[m & lateral] = "film_edge"
        m = b & (zone == "tissue")
        P[m & (loc == FACE_ZMAX)] = "tissue_top"
        P[m & lateral] = "tissue_side"
        P[m & (loc == FACE_ZMIN)] = "tissue_bottom"      # only exists in the tissue-only mesh
        # internal film/tissue interface (needed for coupling and for the porous case)
        zi = self.cell_zone
        inter = self.f_internal & (np.minimum(zi[self.f_owner], zi[np.maximum(self.f_neigh, 0)]) == 2) \
            & (np.maximum(zi[self.f_owner], zi[np.maximum(self.f_neigh, 0)]) == 3)
        self.f_iface = inter
        if np.any(b & (P == "")):
            raise RuntimeError(f"{np.sum(b & (P == ''))} unclassified boundary faces")

    # ------------------------------------------------------------ helpers
    def tissue_nodes(self):
        """Used-node ids of the tissue (levels >= first tissue level)."""
        return np.nonzero(self.node_level >= self.k0[3])[0]

    def iface_node_ids(self):
        """Used-node id of the tissue-bottom node of every plan node (plan order)."""
        K = self.NK + 1
        return self.node_of_all[np.arange(len(self.plan)) * K + self.k0[3]]

    def summary(self):
        z = self.cell_zone
        return dict(points=int(len(self.points)), cells=int(len(self.cells)),
                    **{f"cells_{n}": int(np.sum(z == i)) for i, n in enumerate(ZONES)},
                    plan_nodes=int(len(self.plan)), plan_quads=int(len(self.quads)),
                    min_film_mm=float(self.film_thickness()[~self.node_hole].min()),
                    intrusion_mm=self.intrusion, occluded=self.occluded)

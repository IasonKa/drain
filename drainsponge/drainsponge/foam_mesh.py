"""
Write a WindowMesh straight to an OpenFOAM polyMesh (no gmshToFoam / blockMesh needed)
and check it the way checkMesh does (closed cells, positive volumes, face order,
non-orthogonality, skewness).

Writing the polyMesh ourselves keeps the face order known, so coupling data on the
tissue interface can be written as plain nonuniform lists, and a deformed tissue only
requires a new `points` file (same topology).
"""
from __future__ import annotations
import os
import numpy as np

from .geom import WindowMesh, ZONES

HDR = """FoamFile
{{
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};{note}
}}
"""

# patch order and OpenFOAM types
PATCHES = [("outlet", "patch"), ("film_edge", "patch"), ("tissue_iface", "patch"),
           ("tissue_top", "patch"), ("tissue_side", "patch"),
           ("drain_skin", "wall"), ("standoff", "wall"), ("hole_wall", "wall"), ("wall_under", "wall"),
           ("channel_walls", "wall"), ("tissue_bottom", "wall")]


def _hdr(cls, loc, obj, note=""):
    n = f'\n    note        "{note}";' if note else ""
    return HDR.format(cls=cls, loc=loc, obj=obj, note=n)


class FoamMesh:
    """Face-ordered view of a WindowMesh (internal faces upper-triangular, then patches)."""

    def __init__(self, m: WindowMesh, closed_edge=False):
        """closed_edge: no free liquid at the window edge -> film_edge and tissue_side are walls."""
        self.m = m
        self.closed_edge = closed_edge
        internal = np.nonzero(m.f_internal)[0]
        o = np.lexsort((m.f_neigh[internal], m.f_owner[internal]))
        order = [internal[o]]
        self.patches = []
        start = len(internal)
        for name, typ in PATCHES:
            if closed_edge and name in ("film_edge", "tissue_side"):
                typ = "wall"
            idx = np.nonzero((~m.f_internal) & (m.f_patch == name))[0]
            if len(idx) == 0:
                continue
            idx = idx[np.lexsort((idx, m.f_owner[idx]))]
            order.append(idx)
            self.patches.append(dict(name=name, type=typ, start=start, n=len(idx)))
            start += len(idx)
        self.order = np.concatenate(order)
        self.n_internal = len(internal)
        self.faces = m.f_nodes[self.order]
        self.owner = m.f_owner[self.order]
        self.neigh = m.f_neigh[self.order][: self.n_internal]
        # zone labels
        self.zones = {z: np.nonzero(m.cell_zone == i)[0] for i, z in enumerate(ZONES) if z in m.zones}

    def patch_faces(self, name):
        for p in self.patches:
            if p["name"] == name:
                return np.arange(p["start"], p["start"] + p["n"])
        raise KeyError(name)

    def face_geometry(self, pts_m=None):
        P = self.m.points if pts_m is None else pts_m
        F = P[self.faces]
        S = 0.5 * np.cross(F[:, 2] - F[:, 0], F[:, 3] - F[:, 1])
        C = F.mean(1)
        return C, S

    # -------------------------------------------------------------- writing
    def write(self, case, scale=1e-3):
        d = os.path.join(case, "constant", "polyMesh")
        os.makedirs(d, exist_ok=True)
        self.write_points(case, scale)
        nP, nC, nF = len(self.m.points), len(self.m.cells), len(self.faces)
        note = f"nPoints:{nP}  nCells:{nC}  nFaces:{nF}  nInternalFaces:{self.n_internal}"
        with open(os.path.join(d, "faces"), "w") as f:
            f.write(_hdr("faceList", "constant/polyMesh", "faces"))
            f.write(f"\n{nF}\n(\n")
            f.write("".join(f"4({a} {b} {c} {e})\n" for a, b, c, e in self.faces))
            f.write(")\n")
        for name, arr in (("owner", self.owner), ("neighbour", self.neigh)):
            with open(os.path.join(d, name), "w") as f:
                f.write(_hdr("labelList", "constant/polyMesh", name, note))
                f.write(f"\n{len(arr)}\n(\n")
                f.write("\n".join(map(str, arr)))
                f.write("\n)\n")
        with open(os.path.join(d, "boundary"), "w") as f:
            f.write(_hdr("polyBoundaryMesh", "constant/polyMesh", "boundary"))
            f.write(f"\n{len(self.patches)}\n(\n")
            for p in self.patches:
                extra = "        inGroups        List<word> 1(wall);\n" if p["type"] == "wall" else ""
                f.write(f"    {p['name']}\n    {{\n        type            {p['type']};\n{extra}"
                        f"        nFaces          {p['n']};\n        startFace       {p['start']};\n    }}\n")
            f.write(")\n")
        with open(os.path.join(d, "cellZones"), "w") as f:
            f.write(_hdr("regIOobject", "constant/polyMesh", "cellZones"))
            f.write(f"\n{len(self.zones)}\n(\n")
            for z, ids in self.zones.items():
                f.write(f"{z}\n{{\n    type cellZone;\ncellLabels      List<label> \n{len(ids)}\n(\n")
                f.write("\n".join(map(str, ids)))
                f.write("\n)\n;\n}\n")
            f.write(")\n")
        # film/tissue interface as a faceZone, oriented out of the tissue (-z), so that
        # sum(phi) over it = liquid drawn out of the tissue into the film / hole
        iz = self.iface_faces()
        with open(os.path.join(d, "faceZones"), "w") as f:
            f.write(_hdr("regIOobject", "constant/polyMesh", "faceZones"))
            if len(iz) == 0:
                f.write("\n0\n(\n)\n")
            else:
                C, S = self.face_geometry()
                flip = (S[iz, 2] > 0).astype(int)
                f.write(f"\n1\n(\ntissue_iface_zone\n{{\n    type faceZone;\nfaceLabels      List<label> \n{len(iz)}\n(\n")
                f.write("\n".join(map(str, iz)))
                f.write(f"\n)\n;\nflipMap         List<bool> \n{len(iz)}\n(\n")
                f.write("\n".join(map(str, flip)))
                f.write("\n)\n;\n}\n)\n")

    def iface_faces(self):
        """Face labels (in written order) of the internal film/tissue interface."""
        pos = np.empty(len(self.order), np.int64)
        pos[self.order] = np.arange(len(self.order))
        return np.sort(pos[np.nonzero(self.m.f_iface)[0]])

    def write_points(self, case, scale=1e-3, points=None):
        P = (self.m.points if points is None else points) * scale
        d = os.path.join(case, "constant", "polyMesh")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "points"), "w") as f:
            f.write(_hdr("vectorField", "constant/polyMesh", "points"))
            f.write(f"\n{len(P)}\n(\n")
            f.write("".join(f"({x:.10g} {y:.10g} {z:.10g})\n" for x, y, z in P))
            f.write(")\n")

    # -------------------------------------------------------------- checking
    def check(self, points=None):
        """checkMesh-like report. Raises if the mesh is invalid."""
        P = self.m.points if points is None else points
        nC = len(self.m.cells)
        C, S = self.face_geometry(P)
        own, nei, ni = self.owner, self.neigh, self.n_internal
        # closed cells
        sumS = np.zeros((nC, 3))
        np.add.at(sumS, own, S)
        np.add.at(sumS, nei, -S[:ni])
        magS = np.linalg.norm(S, axis=1)
        sumA = np.zeros(nC)
        np.add.at(sumA, own, magS)
        np.add.at(sumA, nei, magS[:ni])
        open_ = np.linalg.norm(sumS, axis=1) / sumA
        # volumes (divergence theorem about the cell's own centre)
        cc = P[self.m.cells].mean(1)
        vol = np.zeros(nC)
        np.add.at(vol, own, np.einsum("ij,ij->i", C - cc[own], S) / 3)
        np.add.at(vol, nei, -np.einsum("ij,ij->i", C[:ni] - cc[nei], S[:ni]) / 3)
        # face order
        ok_ut = bool(np.all(own[:ni] < nei)) and bool(np.all(np.diff(own[:ni]) >= 0))
        same = np.diff(own[:ni]) == 0
        ok_ut &= bool(np.all(np.diff(nei)[same] > 0))
        # non-orthogonality and skewness of internal faces
        d = cc[nei] - cc[own[:ni]]
        cosang = np.einsum("ij,ij->i", d, S[:ni]) / (np.linalg.norm(d, axis=1) * magS[:ni])
        nonorth = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        # boundary faces must point away from their cell
        bo = own[ni:]
        outward = np.einsum("ij,ij->i", C[ni:] - cc[bo], S[ni:]) > 0
        rep = dict(cells=int(nC), faces=int(len(self.faces)), internal_faces=int(ni),
                   points=int(len(P)), max_open=float(open_.max()),
                   min_volume_mm3=float(vol.min()), total_volume_mm3=float(vol.sum()),
                   upper_triangular=ok_ut, boundary_outward=bool(outward.all()),
                   max_nonorth_deg=float(nonorth.max()), mean_nonorth_deg=float(nonorth.mean()),
                   n_nonorth_gt70=int(np.sum(nonorth > 70)),
                   patches={p["name"]: p["n"] for p in self.patches},
                   zones={z: int(len(i)) for z, i in self.zones.items()})
        bad = []
        if rep["max_open"] > 1e-6:
            bad.append("open cells")
        if rep["min_volume_mm3"] <= 0:
            bad.append("negative volumes")
        if not ok_ut:
            bad.append("face order")
        if not rep["boundary_outward"]:
            bad.append("inward boundary faces")
        if np.any(cosang <= 0):
            bad.append("internal face normal not owner->neighbour")
        rep["ok"] = not bad
        rep["problems"] = bad
        return rep, vol

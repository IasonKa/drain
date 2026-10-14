"""
Exact CAD (OpenCascade via gmsh) of the single-hole window and of the sponge organ.

Window parts (local hole frame, identical to the mesh):
  drain wall    : elliptic skin (8.925 x 4.835) .. skin - wall, Ø1.0 hole (+ optional fillet)
  film          : liquid between skin and tissue (gap g0) + liquid in the hole
  channel       : sheath channel under the wall (outlet at y = +half)
  tissue        : sponge block, bottom = skin + g0, top = skin + g0 + H
Organ (global drain frame): a lobulated sponge organ lying on the drain over the hole,
its underside follows the drain skin at the same gap g0, so it "almost touches" the
drain everywhere and exactly g0 above the modelled hole.
"""
from __future__ import annotations
import json, os
import numpy as np
import gmsh

from .geom import Window, A_OUT, B_OUT


def _start(name):
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.clear()
    gmsh.model.add(name)
    return gmsh.model.occ


def _elliptic_cyl(occ, w: Window, dz, y0, ly, grow=0.0):
    """Elliptic cylinder of the skin, translated by dz along Z, axis along Y."""
    c = occ.addEllipse(w.xc, y0, w.zc + dz, A_OUT + grow, B_OUT + grow, zAxis=[0, 1, 0], xAxis=[1, 0, 0])
    s = occ.addPlaneSurface([occ.addCurveLoop([c])])
    ex = occ.extrude([(2, s)], 0, ly, 0)
    return [d for d in ex if d[0] == 3]


def _box(occ, w: Window, z0, z1, pad=0.0):
    L = w.half + pad
    return [(3, occ.addBox(-L, -L, z0, 2 * L, 2 * L, z1 - z0))]


def _hole(occ, w: Window, z0, z1):
    return [(3, occ.addCylinder(0, 0, z0, 0, 0, z1 - z0, w.a))]


def build_part(occ, w: Window, part):
    L = w.half
    big = (-L - 1, 2 * L + 2)
    zlo, zhi = -w.wall - w.channel_d - 1, w.gap + w.tissue_h + 1
    box = _box(occ, w, zlo, zhi)
    if part == "drain":
        out = _elliptic_cyl(occ, w, 0, *big)
        inn = _elliptic_cyl(occ, w, -w.wall, *big)
        wall, _ = occ.cut(out, inn)
        wall, _ = occ.intersect(wall, box)
        wall, _ = occ.cut(wall, _hole(occ, w, -w.wall - 1, 1))
        if w.fillet > 0:
            occ.synchronize()
            curves = []
            for d, t in gmsh.model.getBoundary(wall, combined=False, oriented=False, recursive=True):
                if d != 1:
                    continue
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, t)
                if zmax > -0.2 * w.wall and max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) < w.a + 1e-3:
                    curves.append(t)
            wall = occ.fillet([v for _, v in wall], sorted(set(curves)), [w.fillet])
        return wall
    if part == "film":
        top = _elliptic_cyl(occ, w, w.gap, *big)
        skin = _elliptic_cyl(occ, w, 0, *big)
        film, _ = occ.cut(top, skin)
        film, _ = occ.intersect(film, box)
        hole_in_wall, _ = occ.intersect(_hole(occ, w, -w.wall, 0.5), _elliptic_cyl(occ, w, 0, *big))
        hole_in_wall, _ = occ.cut(hole_in_wall, _elliptic_cyl(occ, w, -w.wall, *big))
        film, _ = occ.fuse(film, hole_in_wall)
        return film
    if part == "channel":
        a = _elliptic_cyl(occ, w, -w.wall, -L, 2 * L)
        b = _elliptic_cyl(occ, w, -w.wall - w.channel_d, -L - 1, 2 * L + 2)
        ch, _ = occ.cut(a, b)
        ch, _ = occ.intersect(ch, _box(occ, w, zlo, zhi))
        return ch
    if part == "tissue":
        top = _elliptic_cyl(occ, w, w.gap + w.tissue_h, *big)
        bot = _elliptic_cyl(occ, w, w.gap, *big)
        t, _ = occ.cut(top, bot)
        t, _ = occ.intersect(t, box)
        return t
    raise KeyError(part)


def write_window(w: Window, outdir, parts=("drain", "film", "channel", "tissue"), stl=True):
    os.makedirs(outdir, exist_ok=True)
    vols = {}
    for p in parts:
        occ = _start(p)
        dt = build_part(occ, w, p)
        occ.synchronize()
        vols[p] = float(sum(occ.getMass(d, t) for d, t in dt))
        gmsh.write(os.path.join(outdir, f"window_{p}.step"))
        if stl:
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.08)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
            gmsh.model.mesh.generate(2)
            gmsh.write(os.path.join(outdir, f"window_{p}.stl"))
    # one assembly file with all parts
    occ = _start("assembly")
    for p in parts:
        build_part(occ, w, p)
    occ.synchronize()
    gmsh.write(os.path.join(outdir, "window_assembly.step"))
    return vols


# ---------------------------------------------------------------------- organ
def frame_matrix(meta):
    """4x4 local->global from a drainfeb window json (hole_center.json)."""
    X = np.array(meta["global_X_lateral"]); Y = np.array(meta["global_Y_drain_axis"])
    Z = np.array(meta["global_Z_hole_axis"]); O = np.array(meta["global_origin"])
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = X, Y, Z, O
    return T


def build_organ(occ, w: Window, size=(22.0, 30.0, 9.0), seed=4):
    """Lobulated sponge organ resting on the drain (local frame). Returns volume dimTags."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = size
    zc = 0.45 * sz
    s = occ.addSphere(0, 0, zc, 1.0)
    occ.dilate([(3, s)], 0, 0, zc, sx / 2, sy / 2, sz / 2)
    body = [(3, s)]
    lobes = []
    for _ in range(5):                                     # a few lobes for an organic outline
        cx, cy = rng.uniform(-0.35, 0.35) * sx, rng.uniform(-0.4, 0.4) * sy
        r = rng.uniform(0.18, 0.28) * sx
        t = occ.addSphere(cx, cy, zc + rng.uniform(0.0, 0.25) * sz, 1.0)
        occ.dilate([(3, t)], cx, cy, zc, r, r * rng.uniform(1.0, 1.4), 0.38 * sz)
        lobes.append((3, t))
    body, _ = occ.fuse(body, lobes)
    # carve the drain out of it at gap g0: the underside follows the skin
    env = _elliptic_cyl(occ, w, w.gap, -sy, 2 * sy)
    body, _ = occ.cut(body, env)
    # drop pieces that ended up below the drain (keep the largest = the organ)
    if len(body) > 1:
        occ.synchronize()
        m = [occ.getMass(d, t) for d, t in body]
        keep = body[int(np.argmax(m))]
        occ.remove([b for b in body if b != keep], recursive=True)
        body = [keep]
    return body


def write_organ(w: Window, meta, outdir, stl=True):
    os.makedirs(outdir, exist_ok=True)
    T = frame_matrix(meta)
    occ = _start("organ")
    body = build_organ(occ, w)
    occ.synchronize()
    vol_local = float(occ.getMass(*body[0]))
    occ.affineTransform(body, T[:3].ravel().tolist())
    occ.synchronize()
    gmsh.write(os.path.join(outdir, "organ_sponge_global.step"))
    if stl:
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.6)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.05)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 30)
        gmsh.model.mesh.generate(2)
        gmsh.write(os.path.join(outdir, "organ_sponge_global.stl"))
    # window parts in the global frame too, for CAD assembly with the drain
    occ = _start("window_global")
    for p in ("film", "tissue"):
        dt = build_part(occ, w, p)
        occ.affineTransform(dt, T[:3].ravel().tolist())
    occ.synchronize()
    gmsh.write(os.path.join(outdir, "window_film_tissue_global.step"))
    json.dump(dict(local_to_global=T.tolist(), organ_volume_mm3=vol_local,
                   note="local frame: origin on the skin at the hole axis, X lateral, Y drain axis, Z hole axis"),
              open(os.path.join(outdir, "frame.json"), "w"), indent=1)
    return vol_local

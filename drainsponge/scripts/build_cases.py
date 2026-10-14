"""
Build every solver input from one config.

out/openfoam/porous_initial/    Darcy-Brinkman, tissue 50 um above the drain (start of suction)
out/openfoam/porous_sealed/     same, tissue shape after 1 h taken from the preview model
                                (sealed on the rim, bulging into the hole) -> liquid now comes
                                mostly out of the sponge itself
out/openfoam/fluid_template/    channel + hole + film only, tissue = moving boundary (for couple.py)
out/febio/oneway/tissue.feb     FEBio biphasic tissue loaded with the preview's film pressures
                                (one-way test you can run straight away)
out/mesh/*.vtu                  meshes for ParaView

python scripts/build_cases.py [--config cfg.json] [--preview out/preview/preview.json]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.geom import Window, WindowMesh, ZONES
from drainsponge.foam_mesh import FoamMesh
from drainsponge.foam_case import write_case
from drainsponge.febio_model import FebioModel


def write_vtu(m, path):
    import meshio
    meshio.write(path, meshio.Mesh(m.points, [("hexahedron", m.cells)],
                                   cell_data={"zone": [m.cell_zone.astype(np.int32)]}))


def preview_bottom(prev, snap):
    """(r, u_z, p) along the tissue bottom of a preview snapshot."""
    s = prev["snapshots"][snap]
    r = np.array(s["r2"]); u = np.array(s["u"]); p = np.array(s["p_Pa"])
    return r, u[0, :, 1], u[0, :, 0], p[0], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preview", default="out/preview/preview.json")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    cfg = config.load(a.config)
    w = Window.from_dict(cfg["window"])
    od = os.path.join(a.out, "openfoam"); os.makedirs(od, exist_ok=True)
    md = os.path.join(a.out, "mesh"); os.makedirs(md, exist_ok=True)
    w.to_json(os.path.join(a.out, "window.json"))
    json.dump(cfg, open(os.path.join(a.out, "config_used.json"), "w"), indent=1)
    report = {}

    # ---------------------------------------------------------------- porous, initial
    full = WindowMesh(w)
    fm = FoamMesh(full)
    rep, _ = fm.check(); assert rep["ok"], rep
    write_case(fm, os.path.join(od, "porous_initial"), cfg, kind="porous")
    write_vtu(full, os.path.join(md, "window_all_zones.vtu"))
    report["porous_initial"] = rep

    prev = json.load(open(a.preview)) if os.path.exists(a.preview) else None
    # ---------------------------------------------------------------- porous, sealed (1 h)
    if prev:
        r, uz, ur, p, s = preview_bottom(prev, -1)
        m2 = WindowMesh(w)
        x, y = m2.plan.T
        rr = np.hypot(x, y)
        zt = w.skin_e(x) + w.gap + np.interp(rr, r, uz, right=uz[-1])
        m2.set_tissue_bottom(zt)
        fm2 = FoamMesh(m2)
        rep2, _ = fm2.check(); assert rep2["ok"], rep2
        write_case(fm2, os.path.join(od, "porous_sealed"), cfg, kind="porous")
        write_vtu(m2, os.path.join(md, "window_sealed_1h.vtu"))
        report["porous_sealed"] = dict(rep2, from_preview_t=s["t"], **m2.summary())

    # ---------------------------------------------------------------- fluid only (coupling)
    fl = WindowMesh(w, zones=("channel", "hole", "film"))
    ffm = FoamMesh(fl)
    rep3, _ = ffm.check(); assert rep3["ok"], rep3
    write_case(ffm, os.path.join(od, "fluid_template"), cfg, kind="fluid")
    report["fluid_template"] = rep3

    # ---------------------------------------------------------------- FEBio one-way
    tis = WindowMesh(w, zones=("tissue",))
    write_vtu(tis, os.path.join(md, "tissue.vtu"))
    fe = FebioModel(cfg, tis, fl)
    if prev:
        times = [sn["t"] for sn in prev["snapshots"]]
        x, y = tis.plan.T
        rr = np.hypot(x, y)
        P = [np.zeros(len(rr))]
        for k in range(len(times)):
            r, _, _, pb, _ = preview_bottom(prev, k)
            P.append(np.interp(rr, r, pb, right=0.0))
        fe.write(os.path.join(a.out, "febio", "oneway", "tissue.feb"), times, np.array(P))
        report["febio_oneway"] = dict(windows=len(times), t_end=times[-1], nodes=len(tis.points),
                                      hex=len(tis.cells), drain_quads=len(fe.drain_quads))
    json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("ok", "cells") if kk in v} if isinstance(v, dict) else v
                      for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()

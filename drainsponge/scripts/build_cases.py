"""
Build every solver input for one design from one config.

OUT/openfoam/porous_initial/    Darcy-Brinkman, tissue g0 above the drain (start of suction)
OUT/openfoam/porous_sealed/     same, tissue shape after t_end from the preview model
OUT/openfoam/fluid_template/    channel + hole (+ standoff) + film only (for couple.py)
OUT/febio/oneway/tissue.feb     FEBio biphasic tissue loaded with the preview's film pressures
OUT/mesh/*.vtu                  meshes for ParaView

python scripts/build_cases.py --design standoff --film-edge closed [--resolution fine]
                              [--config cfg.json] [--preview PREVIEW.json | --no-preview] [--out DIR]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.geom import Window, WindowMesh
from drainsponge.foam_mesh import FoamMesh
from drainsponge.foam_case import write_case
from drainsponge.febio_model import FebioModel


def write_vtu(m, path):
    import meshio
    meshio.write(path, meshio.Mesh(m.points, [("hexahedron", m.cells)],
                                   cell_data={"zone": [m.cell_zone.astype(np.int32)]}))


def preview_bottom(prev, snap):
    s = prev["snapshots"][snap]
    r = np.array(s["r2"]); u = np.array(s["u"]); p = np.array(s["p_Pa"])
    return r, u[0, :, 1], p[0], s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--design", default=None, choices=list(config.DESIGNS))
    ap.add_argument("--resolution", default=None, choices=list(config.RESOLUTIONS))
    ap.add_argument("--film-edge", default=None, choices=["open", "closed"])
    ap.add_argument("--preview", default=None, help="preview.json (default: OUT/preview/preview.json, computed if missing)")
    ap.add_argument("--no-preview", action="store_true", help="skip everything that needs the preview model")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    over = {"boundary": {"film_edge": a.film_edge}} if a.film_edge else {}
    cfg = config.load(a.config, design=a.design, resolution=a.resolution, **over)
    w = Window.from_dict(cfg["window"])
    closed = cfg.get("boundary", {}).get("film_edge", "open") == "closed"
    od = os.path.join(a.out, "openfoam"); os.makedirs(od, exist_ok=True)
    md = os.path.join(a.out, "mesh"); os.makedirs(md, exist_ok=True)
    w.to_json(os.path.join(a.out, "window.json"))
    json.dump(cfg, open(os.path.join(a.out, "config_used.json"), "w"), indent=1)
    report = {}

    full = WindowMesh(w)
    fm = FoamMesh(full, closed_edge=closed)
    rep, _ = fm.check(); assert rep["ok"], rep
    write_case(fm, os.path.join(od, "porous_initial"), cfg, kind="porous")
    write_vtu(full, os.path.join(md, "window_all_zones.vtu"))
    report["porous_initial"] = dict(rep, **full.summary())

    prev = None
    if not a.no_preview:
        pp = a.preview or os.path.join(a.out, "preview", "preview.json")
        if os.path.exists(pp):
            prev = json.load(open(pp))
        else:
            from drainsponge.preview import Preview
            pv = Preview(cfg)
            snaps_t = list(np.geomspace(5e-4, cfg["time"]["t_end_s"], 9))
            h, snaps = pv.run(t_end=cfg["time"]["t_end_s"], n_steps=80, snap_times=snaps_t, verbose=False)
            prev = dict(snapshots=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in snaps])
    if prev:
        r, uz, p, s = preview_bottom(prev, -1)
        g0_skin = w.top_feature + w.gap
        m2 = WindowMesh(w)
        x, y = m2.plan.T
        rr = np.hypot(x, y)
        m2.set_tissue_bottom(w.skin_e(x) + g0_skin + np.interp(rr, r, uz, right=uz[-1]))
        fm2 = FoamMesh(m2, closed_edge=closed)
        rep2, _ = fm2.check(); assert rep2["ok"], rep2
        write_case(fm2, os.path.join(od, "porous_sealed"), cfg, kind="porous")
        write_vtu(m2, os.path.join(md, "window_deformed_tend.vtu"))
        report["porous_sealed"] = dict(rep2, from_preview_t=s["t"], **m2.summary())

    fl = WindowMesh(w, zones=("channel", "hole", "stand", "film"))
    ffm = FoamMesh(fl, closed_edge=closed)
    rep3, _ = ffm.check(); assert rep3["ok"], rep3
    write_case(ffm, os.path.join(od, "fluid_template"), cfg, kind="fluid")
    report["fluid_template"] = rep3

    tis = WindowMesh(w, zones=("tissue",))
    write_vtu(tis, os.path.join(md, "tissue.vtu"))
    fe = FebioModel(cfg, tis, fl)
    if prev:
        times = [sn["t"] for sn in prev["snapshots"]]
        x, y = tis.plan.T
        rr = np.hypot(x, y)
        Pq = [np.zeros(fe.K)]
        qc = tis.plan[tis.quads].mean(1)
        rq = np.hypot(qc[:, 0], qc[:, 1])
        for k in range(len(times)):
            r, _, pb, _ = preview_bottom(prev, k)
            Pq.append(fe.patch_average(np.interp(rq, r, pb, right=0.0)))
        fe.write(os.path.join(a.out, "febio", "oneway", "tissue.feb"), times, np.array(Pq))
        report["febio_oneway"] = dict(windows=len(times), t_end=times[-1], nodes=len(tis.points),
                                      hex=len(tis.cells), drain_quads=len(fe.drain_quads), patches=fe.K)
    json.dump(report, open(os.path.join(a.out, "build_report.json"), "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("ok", "cells", "patches") if kk in v} for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()

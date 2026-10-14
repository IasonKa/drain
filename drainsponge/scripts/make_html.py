"""
Interactive viewer: out/drainsponge_viewer.html (single file, three.js from a CDN).

3D half-cut of the single-hole window for both designs (current / standoff ring) and both
scenarios (free liquid / no free liquid): drain wall (+ ring), liquid, sponge tissue coloured
by pore pressure and deformed with the reduced-order (preview) results, liquid particles,
time slider, charts, all PNG figures embedded.

python scripts/make_html.py      (after scripts/run_preview.py --compare and the figure scripts)
"""
import base64, glob, json, os, sys
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.geom import Window, WindowMesh, Z_HOLE, Z_STAND, Z_FILM, Z_TIS, Z_CHAN

OUT = "out"


def half_surface(points, cells, keep):
    cl = np.hstack([np.full((keep.sum(), 1), 8), cells[keep]]).ravel()
    g = pv.UnstructuredGrid(cl, np.full(keep.sum(), pv.CellType.HEXAHEDRON), points)
    s = g.extract_surface(algorithm="dataset_surface").triangulate().clean()
    return s.points, s.faces.reshape(-1, 4)[:, 1:]


def r3(a, d=5):
    return np.round(np.asarray(a, float), d).ravel().tolist()


def sig(a, n=3):
    """n significant digits (keeps the JSON small for fields spanning many decades)."""
    return [float(f"{v:.{n}g}") for v in np.asarray(a, float).ravel()]


def geometry(design):
    w = Window.from_dict(config.load(design=design)["window"])
    m = WindowMesh(w)
    P = m.points
    cut = np.pi / 4 if w.has_standoff else 0.0               # cut through two ring arcs, not along a slot
    nrm = np.array([-np.sin(cut), np.cos(cut), 0.0])
    half = P[m.cells].mean(1) @ nrm > 0
    parts = {}
    for name, zones in (("tissue", [Z_TIS]), ("film", [Z_HOLE, Z_STAND, Z_FILM]), ("channel", [Z_CHAN])):
        v, f = half_surface(P, m.cells, half & np.isin(m.cell_zone, zones))
        parts[name] = (v, f)
    K = m.NK + 1
    cells = []
    q = m.quads[~m.cell_hole]
    for k in range(m.k0[1], m.k0[2]):                                   # wall
        lo = q * K + k
        cells.append(np.c_[lo, lo + 1])
    if w.has_standoff:                                                   # ring arcs
        q = m.quads[m.cell_ring]
        for k in range(m.k0[2], m.k0[3]):
            lo = q * K + k
            cells.append(np.c_[lo, lo + 1])
    cells = np.concatenate(cells)
    keep = m.points_all[cells].mean(1) @ nrm > 0
    parts["wall"] = half_surface(m.points_all, cells, keep)
    geo = {n: dict(v=r3(v, 5), f=f.astype(int).ravel().tolist()) for n, (v, f) in parts.items()}
    win = dict(a=w.a, L=w.half, gap=w.gap, hs=w.top_feature, H=w.tissue_h, wall=w.wall, xc=w.xc, zc=w.zc,
               A=8.925, B=4.835, rin=w.standoff_rin, rout=w.standoff_rout, cut=cut,
               slots=[float(np.arctan2(d[1], d[0])) for d in w.slot_dirs()] if w.has_standoff else [])
    return geo, win, {n: len(g["f"]) // 3 for n, g in geo.items()}


def run_data(path, mu):
    prev = json.load(open(path))
    zsel = slice(None, None, 2)                                          # every 2nd row in z keeps the file small
    s0 = prev["snapshots"][0]
    r2, z2 = np.array(s0["r2"]), np.array(s0["z2"])[zsel]
    snaps = []
    for s in prev["snapshots"]:
        u, p = np.array(s["u"])[zsel], np.array(s["p_Pa"])[zsel]
        xc, wv = np.array(s["xc"]), np.array(s["w_mm_s"])
        R, Z = np.meshgrid(r2, z2)
        wr = griddata(xc, wv[:, 0], (R, Z), method="linear")
        wz = griddata(xc, wv[:, 1], (R, Z), method="linear")
        wr = np.where(np.isnan(wr), griddata(xc, wv[:, 0], (R, Z), method="nearest"), wr)
        wz = np.where(np.isnan(wz), griddata(xc, wv[:, 1], (R, Z), method="nearest"), wz)
        hf = np.array(s["film_h"]); fp = np.array(s["film_p_Pa"])
        vfilm = -(hf ** 2) / (12 * mu) * np.gradient(fp * 1e-6, r2)
        snaps.append(dict(t=s["t"], ph=s["p_hole_Pa"], ur=sig(u[..., 0]), uz=sig(u[..., 1]),
                          p=sig(p, 3), wr=sig(wr, 2), wz=sig(wz, 2), h=sig(hf, 3), vf=sig(vfilm, 3)))
    h = prev["history"]
    seep = np.array(h["Q_body"]) + np.array(h["Q_seep"])
    hist = dict(t=r3(h["t"], 9), Q=r3(np.abs(h["Q_hole"]) * 0.06, 9), Qf=r3(np.clip(h["Q_edge"], 0, None) * 0.06, 9),
                Qs=r3(np.where(seep > 0, seep, 0) * 0.06, 12), cl=r3(np.array(h["clearance"]) * 1e3, 2))
    g = prev["geometry"]
    zb = np.where((r2 >= g["rin"] - 1e-9) & (r2 <= g["rout"] + 1e-9), g["hs"], 0.0) if g["hs"] > 0 else np.zeros(len(r2))
    return dict(r2=r3(r2, 6), z2=r3(z2, 6), zb=r3(zb, 4), snaps=snaps, hist=hist, summary=prev["summary"],
                g0=g["g0"], R=g["R"], open=g["film_edge_open"],
                params=dict(E=prev["config"]["tissue"]["E_Pa"], dp=prev["config"]["suction"]["dp_Pa"],
                            k=prev["config"]["tissue"]["perm_m4_Ns"]))


def main():
    cfg = config.load()
    mu = cfg["fluid"]["mu_Pa_s"] * 1e-6
    runs = {}
    for p in sorted(glob.glob(os.path.join(OUT, "preview", "*", "preview.json"))):
        runs[os.path.basename(os.path.dirname(p))] = run_data(p, mu)
    designs, sizes = {}, {}
    for d in config.DESIGNS:
        if any(k.startswith(d + "_") and k[len(d) + 1:] in ("open", "closed") for k in runs):
            geo, win, n = geometry(d)
            designs[d] = dict(geo=geo, win=win)
            sizes[d] = n
    data = dict(designs=designs, runs=runs)
    imgs = []
    order = [("01_drain_with_sponge_organ_standoff.png", "New standoff design with the sponge organ on it"),
             ("04_section_at_hole_standoff.png", "Section through the studied hole, standoff design (real CAD)"),
             ("02_hole_window_cutaway_standoff.png", "Modelled single-hole window, standoff design"),
             ("02_hole_window_cutaway_current.png", "Modelled single-hole window, current design"),
             ("02_hole_window_cutaway_standoff_opt.png", "Modelled single-hole window, optimized ring"),
             ("03_mesh_section_standoff.png", "Shared hex mesh (OpenFOAM liquid + FEBio tissue), standoff"),
             ("09_design_comparison.png", "Current vs standoff: tissue position and flows over time"),
             ("08_standoff_screening.png", "Standoff ring screening: height, width, slot"),
             ("05_timeline_standoff_closed.png", "Standoff, no free liquid: time history"),
             ("06_fields_standoff_closed.png", "Standoff, no free liquid: pressure and flow in the sponge"),
             ("05_timeline_current_closed.png", "Current design, no free liquid: time history"),
             ("06_fields_current_closed.png", "Current design, no free liquid: pressure and flow in the sponge"),
             ("05_timeline_standoff_open.png", "Standoff, free liquid: time history"),
             ("05_timeline_current_open.png", "Current design, free liquid: time history"),
             ("07_parameter_map.png", "Current design: stiffness x suction map"),
             ("01_drain_with_sponge_organ_current.png", "Current dome-cap design with the sponge organ"),
             ("04_section_at_hole_current.png", "Section through the studied hole, current design")]
    for fn, cap in order:
        pth = os.path.join(OUT, "figures", fn)
        if os.path.exists(pth):
            from io import BytesIO
            from PIL import Image
            im = Image.open(pth).convert("RGB")
            if im.width > 1400:
                im = im.resize((1400, round(im.height * 1400 / im.width)), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, "JPEG", quality=85, optimize=True)
            imgs.append(dict(src="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), cap=cap))
    html = open(os.path.join(os.path.dirname(__file__), "viewer_template.html"), encoding="utf-8").read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("/*__IMGS__*/[]", json.dumps(imgs))
    out = os.path.join(OUT, "drainsponge_viewer.html")
    open(out, "w", encoding="utf-8").write(html)
    print(out, f"{os.path.getsize(out)/1e6:.1f} MB", sizes, list(runs))


if __name__ == "__main__":
    main()

"""
Interactive viewer: out/drainsponge_viewer.html (single file, three.js from a CDN).

3D half-cut of the single-hole window: drain wall, liquid film / hole, sponge tissue
coloured by pore pressure and deformed with the reduced-order (preview) results,
liquid particles moving inside the sponge and in the film; time slider; charts;
all PNG figures embedded.

python scripts/make_html.py
"""
import base64, json, os, sys
import numpy as np
import pyvista as pv
from scipy.interpolate import griddata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.geom import Window, WindowMesh

OUT = "out"


def half_surface(points, cells, keep):
    cl = np.hstack([np.full((keep.sum(), 1), 8), cells[keep]]).ravel()
    g = pv.UnstructuredGrid(cl, np.full(keep.sum(), pv.CellType.HEXAHEDRON), points)
    s = g.extract_surface(algorithm="dataset_surface").triangulate().clean()
    return s.points, s.faces.reshape(-1, 4)[:, 1:]


def r3(a, d=5):
    return np.round(np.asarray(a, float), d).ravel().tolist()


def main():
    cfg = json.load(open(os.path.join(OUT, "config_used.json"))) if os.path.exists(os.path.join(OUT, "config_used.json")) else config.load()
    w = Window.from_dict(cfg["window"])
    m = WindowMesh(w)
    P = m.points
    cy = P[m.cells].mean(1)[:, 1]
    half = cy > 0
    parts = {}
    for name, zones in (("tissue", [3]), ("film", [1, 2]), ("channel", [0])):
        k = half & np.isin(m.cell_zone, zones)
        v, f = half_surface(P, m.cells, k)
        parts[name] = (v, f)
    # drain wall half: hexes between wall bottom and skin outside the hole
    K = m.NK + 1
    q = m.quads[~m.cell_hole]
    cells = []
    for k in range(m.k0[1], m.k0[2]):
        lo = q * K + k
        cells.append(np.c_[lo, lo + 1])
    cells = np.concatenate(cells)
    keep = m.points_all[cells].mean(1)[:, 1] > 0
    parts["wall"] = half_surface(m.points_all, cells, keep)

    prev = json.load(open(os.path.join(OUT, "preview", "preview.json")))
    g0 = w.gap
    snaps = []
    for s in prev["snapshots"]:
        r2, z2 = np.array(s["r2"]), np.array(s["z2"])
        u, p = np.array(s["u"]), np.array(s["p_Pa"])
        xc, wv = np.array(s["xc"]), np.array(s["w_mm_s"])
        R, Z = np.meshgrid(r2, z2)
        wr = griddata(xc, wv[:, 0], (R, Z), method="linear")
        wz = griddata(xc, wv[:, 1], (R, Z), method="linear")
        wr = np.where(np.isnan(wr), griddata(xc, wv[:, 0], (R, Z), method="nearest"), wr)
        wz = np.where(np.isnan(wz), griddata(xc, wv[:, 1], (R, Z), method="nearest"), wz)
        hf = np.array(s["film_h"]); fp = np.array(s["film_p_Pa"])
        mu = cfg["fluid"]["mu_Pa_s"] * 1e-6
        vfilm = -(hf ** 2) / (12 * mu) * np.gradient(fp * 1e-6, r2)          # mean film speed (mm/s)
        snaps.append(dict(t=s["t"], ph=s["p_hole_Pa"], ur=r3(u[..., 0], 6), uz=r3(u[..., 1], 6),
                          p=r3(p, 2), wr=r3(wr, 12), wz=r3(wz, 12), h=r3(hf, 6), vf=r3(vfilm, 6)))
    h = prev["history"]
    seep = (np.array(h["Q_body"]) + np.array(h["Q_seep"]))
    hist = dict(t=r3(h["t"], 9), Q=r3(np.abs(h["Q_hole"]) * 0.06, 9), Qf=r3(np.clip(h["Q_edge"], 0, None) * 0.06, 9),
                Qs=r3(np.where(seep > 0, seep, 0) * 0.06, 12), hr=r3(np.array(h["h_rim"]) * 1e3, 3),
                di=r3(np.array(h["intrusion"]) * 1e3, 3))
    geo = {}
    for n, (v, f) in parts.items():
        geo[n] = dict(v=r3(v, 5), f=f.astype(int).ravel().tolist())
    x = None
    data = dict(geo=geo, r2=r3(prev["snapshots"][0]["r2"], 6), z2=r3(prev["snapshots"][0]["z2"], 6),
                snaps=snaps, hist=hist, summary=prev["summary"],
                win=dict(a=w.a, L=w.half, g0=g0, H=w.tissue_h, wall=w.wall, xc=w.xc, zc=w.zc, A=8.925, B=4.835,
                         dp=cfg["suction"]["dp_Pa"], E=cfg["tissue"]["E_Pa"], k=cfg["tissue"]["perm_m4_Ns"],
                         R=prev["geometry"]["R"]))
    imgs = []
    for fn, cap in (("01_drain_with_sponge_organ.png", "Dome-cap drain with the sponge organ lying on it"),
                    ("04_section_at_hole.png", "Cross-section through the studied hole (real drain geometry)"),
                    ("02_hole_window_cutaway.png", "The modelled single-hole window"),
                    ("03_mesh_section.png", "Shared hex mesh: OpenFOAM fluid + FEBio tissue"),
                    ("05_preview_timeline.png", "Time history: film flow, seal, liquid out of the sponge"),
                    ("06_preview_fields.png", "Pore pressure and liquid flow inside the sponge"),
                    ("07_parameter_map.png", "Sensitivity: stiffness, suction, permeability")):
        pth = os.path.join(OUT, "figures", fn)
        if os.path.exists(pth):
            imgs.append(dict(src="data:image/png;base64," + base64.b64encode(open(pth, "rb").read()).decode(), cap=cap))
    html = open(os.path.join(os.path.dirname(__file__), "viewer_template.html"), encoding="utf-8").read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("/*__IMGS__*/[]", json.dumps(imgs))
    out = os.path.join(OUT, "drainsponge_viewer.html")
    open(out, "w", encoding="utf-8").write(html)
    print(out, f"{os.path.getsize(out)/1e6:.1f} MB", {k: len(v['f']) // 3 for k, v in geo.items()})


if __name__ == "__main__":
    main()

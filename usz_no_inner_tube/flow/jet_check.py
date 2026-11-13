"""Jet quantities of the blocked-sheath run for comparison with the analytical jet model.

  python jet_check.py runs/blocked   ->  results/blocked/jet_check.json
Per cross-section at distance x from the irrigation outlet: centre-line speed, forward flow (towards suction),
backward flow (towards the tip), jet half-width.
"""
import json, os, sys
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
XC, ZC, YB = 0.78, -4.82, 59.9
run = sys.argv[1]
name = os.path.basename(os.path.normpath(run))
m = np.load(os.path.join(run, "mesh.npz"))
r = np.load(os.path.join(run, "result.npz"))
xyz, tets = m["xyz"], m["tets"]
g = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                        np.full(len(tets), pv.CellType.TETRA), xyz)
g["v"] = r["v"] * 1e3                                          # mm/s
g["w"] = -g["v"][:, 1]                                         # speed towards suction, mm/s

xs = np.r_[0.5, 1, 2, 3, 4, 6, 8, 10:119:2.0]
rows = []
for x in xs:
    s = g.slice(normal=(0, 1, 0), origin=(0, YB - x, 0)).triangulate()
    s = s.compute_cell_sizes(length=False, area=True, volume=False).point_data_to_cell_data()
    a, w = np.asarray(s["Area"]), np.asarray(s["w"])
    qf = float((a * np.clip(w, 0, None)).sum()) * 60e-3          # mL/min
    qb = float((a * np.clip(-w, 0, None)).sum()) * 60e-3
    # centre line: the jet axis may wander a little, take the maximum within 1.5 mm of the axis
    c = s.cell_centers().points
    rad = np.hypot(c[:, 0] - XC, c[:, 2] - ZC)
    near = rad < 1.5
    uc = float(w[near].max())
    # half-width: area where w > uc/2, as an equivalent radius (only meaningful while the jet is distinct)
    rh = float(np.sqrt(a[(w > 0.5 * uc) & (rad < 6)].sum() / np.pi))
    rows.append(dict(x=float(x), uc=uc, q_forward=qf, q_back=qb, r_half=rh, area=float(a.sum())))
    print(f"x={x:6.1f} uc={uc:8.2f} mm/s  forward={qf:7.2f}  back={qb:7.2f} mL/min  r_half={rh:5.2f} mm")
out = os.path.join(HERE, "results", name, "jet_check.json")
json.dump(rows, open(out, "w"), indent=1)
print("written", out)

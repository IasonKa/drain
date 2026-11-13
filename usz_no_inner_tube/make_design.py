"""
USZ drain sheath WITHOUT the internal (spoked) tube, with a V (funnel) connection to the suction tube.

Takes the current USZ CAD (others/drainage-3D-USZ-Vis/drainage-3D.stl) and
  - removes the internal spoked tube over the whole sheath,
  - cuts the round suction tube back to the narrow end of the funnel collar, so the collar becomes an
    open V (funnel) from the sheath lumen into the suction tube,
  - keeps the sheath, its ribs, the V-ribs and every hole unchanged,
  - adds a tip cap with the irrigation port (d 1.0 mm bore, 1.6 mm outside) on the axis.

  python make_design.py
Writes usz_sheath_v_connection.stl (sheath + funnel + suction tube), irrigation_tip_cap.stl,
removed_internal_tube.stl. Local frame (same as drainsponge): origin on the skin at the studied hole,
X lateral, Y along the drain (tip at +Y), Z along the hole axis.
"""
import json, os
import numpy as np
import pyvista as pv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
SRC = os.path.join(ROOT, "others", "drainage-3D-USZ-Vis", "drainage-3D.stl")
META = os.path.join(HERE, "..", "drainsponge", "data", "hole_center.json")

XC, ZC = 0.78, -4.82        # axis of the internal tube / suction tube in the local frame (mm)
R_CUT = 2.95                # internal tube outer radius 2.77; V-rib tips are at 3.12
Y0, Y1 = -59.5, 60.5        # sheath runs from the funnel (Y0) to the tip (Y1)
A_OUT, B_OUT = 8.925, 4.835 # skin ellipse half-axes
CAP_T = 0.6                 # tip cap thickness
R_PORT_IN, R_PORT_OUT = 0.5, 0.8   # irrigation port bore / outside radius
PORT_INSIDE, PORT_OUTSIDE = 1.0, 4.0   # how far the port sticks into the lumen / out of the cap


def local_frame():
    m = json.load(open(META))
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = m["global_X_lateral"], m["global_Y_drain_axis"], m["global_Z_hole_axis"]
    T[:3, 3] = m["global_origin"]
    return np.linalg.inv(T)


def load_local():
    return pv.read(SRC).transform(local_frame(), inplace=False).clean()


def _surf(m):
    return m.extract_surface(algorithm="dataset_surface").triangulate()


def _ring(y, r_in, r_out_fn, n=96, flip=False):
    """Flat ring in the plane Y = y between a circle r_in and an outer outline r_out_fn(theta) -> (x, z)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    inner = np.c_[XC + r_in * np.cos(th), np.full(n, y), ZC + r_in * np.sin(th)]
    ox, oz = r_out_fn(th)
    outer = np.c_[ox, np.full(n, y), oz]
    i = np.arange(n)
    j = (i + 1) % n
    f = np.r_[np.c_[i, j, n + j], np.c_[i, n + j, n + i]]
    if flip:
        f = f[:, ::-1]
    return pv.PolyData(np.r_[inner, outer], np.c_[np.full(len(f), 3), f].ravel())


def _shell(ya, yb, fn, n=96):
    """Side wall between Y = ya and yb along the outline fn(theta) -> (x, z)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, z = fn(th)
    a = np.c_[x, np.full(n, ya), z]
    b = np.c_[x, np.full(n, yb), z]
    i = np.arange(n)
    j = (i + 1) % n
    f = np.r_[np.c_[i, j, n + j], np.c_[i, n + j, n + i]]
    return pv.PolyData(np.r_[a, b], np.c_[np.full(len(f), 3), f].ravel())


def circle(r):
    return lambda th: (XC + r * np.cos(th), ZC + r * np.sin(th))


def ellipse(th):
    return XC + A_OUT * np.cos(th), ZC + B_OUT * np.sin(th)


def build():
    cur = load_local()
    p = cur.points
    r = np.hypot(p[:, 0] - XC, p[:, 2] - ZC)
    inside = (r < R_CUT) & (p[:, 1] < Y1 + 0.2)           # internal tube + suction tube (same cylinder)
    tri = cur.faces.reshape(-1, 4)[:, 1:]
    core = inside[tri].all(axis=1)
    sheath = _surf(cur.extract_cells(np.nonzero(~core)[0]))
    tube = _surf(cur.extract_cells(np.nonzero(core)[0]))
    # narrow end of the funnel collar: where its inner cone lands on the suction tube
    sp = sheath.points
    sr = np.hypot(sp[:, 0] - XC, sp[:, 2] - ZC)
    on_tube = (sp[:, 1] < Y0 - 1) & (sr < R_CUT)
    y_cut = float(sp[on_tube, 1].max())
    tp = tube.points
    tr = np.hypot(tp[:, 0] - XC, tp[:, 2] - ZC)
    far = tp[:, 1] < y_cut - 2
    r_in, r_out = 2.1, float(np.median(tr[far & (tr > 2.5)]))   # bore 4.2 mm (lengthwise section of the CAD)
    suction = tube.clip(normal=(0, 1, 0), origin=(0, y_cut, 0), invert=True).triangulate()
    removed = tube.clip(normal=(0, 1, 0), origin=(0, y_cut, 0), invert=False).triangulate()
    end_face = _ring(y_cut, r_in, circle(r_out))
    body = _surf(pv.merge([sheath, suction, end_face], merge_points=False))
    # tip cap with the irrigation port
    cap = pv.merge([_ring(Y1, R_PORT_OUT, ellipse, flip=True), _ring(Y1 + CAP_T, R_PORT_OUT, ellipse),
                    _shell(Y1, Y1 + CAP_T, ellipse),
                    _shell(Y1 - PORT_INSIDE, Y1 + PORT_OUTSIDE, circle(R_PORT_OUT)),
                    _shell(Y1 - PORT_INSIDE, Y1 + PORT_OUTSIDE, circle(R_PORT_IN)),
                    _ring(Y1 - PORT_INSIDE, R_PORT_IN, circle(R_PORT_OUT), flip=True),
                    _ring(Y1 + PORT_OUTSIDE, R_PORT_IN, circle(R_PORT_OUT))], merge_points=False)
    cap = _surf(cap)
    tip_tris = sheath.extract_cells(np.nonzero((sp[sheath.faces.reshape(-1, 4)[:, 1:], 1] > Y1 - 0.1).all(axis=1))[0])
    info = dict(y_funnel_wide=Y0, y_funnel_narrow=round(y_cut, 2), funnel_length=round(Y0 - y_cut, 2),
                suction_tube_bore=round(2 * r_in, 2), suction_tube_outside=round(2 * r_out, 2),
                sheath_length=round(Y1 - Y0, 1), original_tip_face_area_mm2=round(float(_surf(tip_tris).area), 1),
                skin_ellipse_area_mm2=round(float(np.pi * A_OUT * B_OUT), 1),
                triangles=dict(body=body.n_cells, removed=removed.n_cells, cap=cap.n_cells))
    return dict(current=cur, body=body, removed=removed, cap=cap, info=info)


if __name__ == "__main__":
    d = build()
    print(json.dumps(d["info"], indent=1))
    d["body"].save(os.path.join(HERE, "usz_sheath_v_connection.stl"))
    d["cap"].save(os.path.join(HERE, "irrigation_tip_cap.stl"))
    d["removed"].save(os.path.join(HERE, "removed_internal_tube.stl"))
    json.dump(d["info"], open(os.path.join(HERE, "design_info.json"), "w"), indent=1)

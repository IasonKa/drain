"""Interactive 3D page of the design: python export_viewer.py  ->  viewer.html (from viewer_template.html)"""
import base64, json, os
import numpy as np
from make_design import build, HERE, XC, ZC

d = build()
json.dump(d["info"], open(os.path.join(HERE, "design_info.json"), "w"), indent=1)
d["body"].save(os.path.join(HERE, "usz_sheath_v_connection.stl"))
d["cap"].save(os.path.join(HERE, "irrigation_tip_cap.stl"))
d["removed"].save(os.path.join(HERE, "removed_internal_tube.stl"))


def pack(m):
    m = m.clean().triangulate()
    pts = (m.points - np.array([XC, 0, ZC])).astype("<f4")
    idx = m.faces.reshape(-1, 4)[:, 1:]
    it = "<u2" if len(pts) < 65536 else "<u4"
    return dict(p=base64.b64encode(pts.tobytes()).decode(), i=base64.b64encode(idx.astype(it).tobytes()).decode(),
                w=2 if it == "<u2" else 4)


data = dict(body=pack(d["body"]), removed=pack(d["removed"]), cap=pack(d["cap"]), info=d["info"])
html = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
html = html.replace("/*DATA*/null", json.dumps(data))
open(os.path.join(HERE, "viewer.html"), "w", encoding="utf-8").write(html)
print(json.dumps(d["info"], indent=1))
print("viewer.html %.1f MB" % (len(html) / 1e6))

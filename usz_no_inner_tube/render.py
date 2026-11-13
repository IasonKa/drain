"""Pictures of the sheath without the internal tube (run make_design.py first)."""
import os, sys
import numpy as np
import pyvista as pv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_design import HERE, load_local, XC, ZC, Y0

GREY, ORANGE = "#8a9bb0", "#e08a2e"
cur = load_local()
new = pv.read(os.path.join(HERE, "usz_sheath_no_inner_tube.stl"))

def seg(mesh, y0, y1):
    return mesh.clip(normal=(0, 1, 0), origin=(0, y0, 0), invert=False).clip(normal=(0, 1, 0), origin=(0, y1, 0), invert=True)

# 1: whole part, cut open lengthwise (top half removed), current vs new
p = pv.Plotter(off_screen=True, window_size=(2000, 1300), shape=(2, 1))
for i, (m, t) in enumerate(((cur, "current USZ design: sheath + internal spoked tube"),
                            (new, "new: sheath without the internal tube (short stub kept at the connector)"))):
    p.subplot(i, 0)
    h = seg(m, -75, 62).clip(normal=(0, 0, 1), origin=(0, 0, ZC + 0.9), invert=True)
    p.add_mesh(h, color=GREY, smooth_shading=False, specular=0.3)
    p.add_text(t, font_size=13)
    p.camera_position = [(XC + 60, -150, 110), (XC, -6, ZC), (0, 0, 1)]
    p.camera.zoom(1.5)
p.screenshot(os.path.join(HERE, "01_cut_open_current_vs_new.png")); p.close()

# 2: connector end close-up, cut open
p = pv.Plotter(off_screen=True, window_size=(2000, 1000), shape=(1, 2))
for i, (m, t) in enumerate(((cur, "current"), (new, "new: stub of the internal tube at the connector"))):
    p.subplot(0, i)
    h = seg(m, -75, -35).clip(normal=(0, 0, 1), origin=(0, 0, ZC + 0.9), invert=True)
    p.add_mesh(h, color=GREY, specular=0.3)
    p.add_text(t, font_size=13)
    p.camera_position = [(XC + 10, -20, 22), (XC, -52, ZC), (0, 0, 1)]
    p.camera.zoom(1.0)
p.screenshot(os.path.join(HERE, "02_connector_end_closeup.png")); p.close()

# 3: look into the sheath from a cut face
p = pv.Plotter(off_screen=True, window_size=(2000, 1000), shape=(1, 2))
for i, (m, t) in enumerate(((cur, "current"), (new, "new: open lumen, V-ribs and sheath ribs kept"))):
    p.subplot(0, i)
    p.add_mesh(seg(m, -75, 1.1), color=GREY, specular=0.3)
    p.add_text(t, font_size=13)
    p.camera_position = [(XC + 14, 34, ZC + 12), (XC, 0, ZC), (0, 0, 1)]
    p.camera.zoom(1.0)
p.screenshot(os.path.join(HERE, "03_view_into_sheath.png")); p.close()

# 4: cross-sections
def loops(mesh, y):
    sl = mesh.slice(normal=(0, 1, 0), origin=(0, y, 0)).strip(join=True)
    L, lines, i = [], sl.lines, 0
    while i < len(lines):
        k = lines[i]; L.append(sl.points[lines[i + 1:i + 1 + k]][:, [0, 2]]); i += k + 1
    return L
fig, axs = plt.subplots(2, 3, figsize=(19, 8.4))
stations = [(1.1, "between holes (y = 1.1 mm)"), (-20.0, "through a row of holes (y = -20 mm)"), (Y0 + 3, "in the stub (3 mm from the connector)")]
for r, (m, name) in enumerate(((cur, "current"), (new, "new"))):
    for c, (y, lab) in enumerate(stations):
        ax = axs[r, c]
        for l in loops(m, y):
            inner = np.hypot(l[:, 0] - XC, l[:, 1] - ZC).max() < 3.0
            ax.fill(l[:, 0], l[:, 1], fc=ORANGE if inner else GREY, ec="#3d4a5c", lw=0.6, alpha=0.9 if inner else 0.35)
            ax.plot(np.r_[l[:, 0], l[0, 0]], np.r_[l[:, 1], l[0, 1]], color="#3d4a5c", lw=0.9)
        ax.set_aspect("equal"); ax.set_xlim(-9, 10.6); ax.set_ylim(-10.3, 0.7)
        ax.set_title(f"{name}: {lab}", fontsize=11); ax.set_xlabel("lateral X (mm)"); ax.set_ylabel("Z (mm)")
axs[0, 0].annotate("internal tube\n(removed)", xy=(XC + 2.4, ZC + 1.6), xytext=(5.2, -1.2), fontsize=10, arrowprops=dict(arrowstyle="->"))
axs[1, 0].annotate("V-rib kept", xy=(XC, -1.2), xytext=(XC - 1.3, -4.3), fontsize=10, arrowprops=dict(arrowstyle="->"))
axs[1, 0].annotate("", xy=(XC, -8.4), xytext=(XC, -5.0), arrowprops=dict(arrowstyle="->"))
fig.suptitle("USZ sheath: outlines are material boundaries; orange = internal tube", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "04_cross_sections.png"), dpi=110)

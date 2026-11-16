"""Figures for final_report.tex: design render, marker maps of the old fluid, sweep graph copy."""
import json, os, shutil
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "flow", "results")
YA, YB = -59.5, 59.9
shutil.copy(os.path.join(RES, "sweep.png"), os.path.join(HERE, "fig_sweep.png"))

# ---- old fluid markers: where they started, coloured by whether they are still inside after 10 min
cases = [("blocked", "outlet at the tip"), ("s030", "outlet 30 mm from the tip"), ("s060", "outlet 60 mm from the tip")]
fig, axs = plt.subplots(len(cases), 1, figsize=(7.4, 4.3), sharex=True)
for ax, (c, ttl) in zip(axs, cases):
    m = np.load(os.path.join(RES, c, "flush_markers.npz"))
    s = json.load(open(os.path.join(RES, c, "flush.json")))["s_mm"]
    x = YB - m["seeds"][:, 1]; lat = m["seeds"][:, 0]
    left = m["t_exit"] > 600
    ax.scatter(x[~left], lat[~left], s=1.5, color="#9ecbff", lw=0, label="flushed out within 10 min")
    ax.scatter(x[left], lat[left], s=1.5, color="#d1242f", lw=0, label="still inside after 10 min")
    ax.axvline(s, color="k", lw=1); ax.text(s + 1, lat.max() * 0.72, "outlet", fontsize=7)
    ax.set_title(f"{ttl}: {100 * left.mean():.0f} % of the old fluid left", fontsize=9, loc="left")
    ax.set_ylabel("across (mm)"); ax.set_aspect("equal"); ax.set_xlim(-1, 121)
axs[-1].set_xlabel("distance from the tip (mm); suction is on the right")
axs[0].legend(frameon=False, fontsize=7, markerscale=5, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.45))
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_markers.png"), dpi=200); plt.close(fig)

# ---- design render (cut open along the wide mid-plane, tip on the left)
pl = pv.Plotter(off_screen=True, window_size=(2400, 620)); pl.set_background("white")
ZC = -4.82
for f, col in (("usz_sheath_v_connection.stl", "#c9d6e4"), ("irrigation_tip_cap.stl", "#35a7ff")):
    f = os.path.join(ROOT, f)
    if os.path.exists(f):
        mesh = pv.read(f).clip(normal=(0, 1, 0), origin=(0, -86, 0), invert=False).clip(normal=(0, 0, 1), origin=(0, 0, ZC), invert=True).clip(normal=(0, 0, 1), origin=(0, 0, -13), invert=False)
        pl.add_mesh(mesh, color=col, smooth_shading=True, split_sharp_edges=True)
pl.camera_position = [(0.8, -12, 300), (0.8, -12, ZC), (-1, 0, 0)]
pl.enable_parallel_projection(); pl.camera.parallel_scale = 20
pl.screenshot(os.path.join(HERE, "fig_design.png")); pl.close()

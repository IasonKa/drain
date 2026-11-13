"""Drawing of what the flow runs simulate (no solver, no mesh): python show_setup.py"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from make_design import load_local, XC, ZC

YA, YB, LT, R_TUBE, R_NOZ, LN = -59.5, 59.9, 12.0, 2.1, 0.5, 2.0
A_OUT, B_OUT = 8.925, 4.835
GROUPS = {"tip": ([(0.0, 54.0), (3.3, 54.0), (0.0, 49.5), (3.3, 49.5)], "#d62728"),
          "mid": ([(0.0, 0.0), (3.3, 0.0), (0.0, 4.5), (3.3, 4.5)], "#2ca02c"),
          "conn": ([(0.0, -54.0), (3.3, -54.0), (0.0, -49.5), (3.3, -49.5)], "#9467bd")}
FL, WALL, IRR = "#cfe6fb", "#8a9bb0", "#1f77b4"

sl = load_local().slice(normal=(0, 1, 0), origin=(0, 1.1, 0)).strip(join=True)
loops, lines, i = [], sl.lines, 0
while i < len(lines):
    k = lines[i]
    P = sl.points[lines[i + 1:i + 1 + k]][:, [0, 2]]
    i += k + 1
    if np.hypot(P[:, 0] - XC, P[:, 1] - ZC).max() > 5:
        loops.append(P)
area = lambda P: 0.5 * abs(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))
loops = sorted(loops, key=area)
skin, lumen = loops[-1], loops[-2]
print("lumen area %.1f mm2, skin area %.1f mm2" % (area(lumen), area(skin)))

fig = plt.figure(figsize=(19, 9.5))
gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1], height_ratios=[1, 1])
# ---- top view
ax = fig.add_subplot(gs[0, 0])
ax.add_patch(Rectangle((YA, XC - A_OUT), YB - YA + 0.6, 2 * A_OUT, fc=WALL, ec="k", lw=0.8))
ax.add_patch(Rectangle((YA, XC - A_OUT + 0.6), YB - YA, 2 * A_OUT - 1.2, fc=FL, ec="none"))
ax.add_patch(Rectangle((YA - LT, XC - R_TUBE), LT, 2 * R_TUBE, fc=FL, ec="k", lw=0.8))
ax.add_patch(Rectangle((YB, XC - R_NOZ), LN + 0.6, 2 * R_NOZ, fc=IRR, ec="k", lw=0.6))
ax.annotate("", xy=(YB - 14, XC), xytext=(YB - 1, XC), arrowprops=dict(arrowstyle="->", color=IRR, lw=2.5))
ax.text(YB - 1, XC + 1.6, "irrigation outlet d = 1.0 mm\n250 mL/h, aimed at the suction end", ha="right", fontsize=10, color=IRR)
ax.annotate("", xy=(YA - LT - 5, XC), xytext=(YA - LT, XC), arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
ax.text(YA - LT - 5, XC + 3.2, "suction\n-7 mmHg", ha="center", fontsize=10)
for name, (hs, c) in GROUPS.items():
    for x, y in hs:
        ax.add_patch(Circle((y, x), 0.5, fc=c, ec="k", lw=0.5))
    ax.text(np.mean([h[1] for h in hs]), XC - A_OUT - 2.2, f'run "{name}": these 4 holes open', ha="center", color=c, fontsize=10)
ax.text(0, XC + A_OUT + 1.2, 'run "blocked": no hole open (tissue covers the whole sheath)', ha="center", fontsize=10)
ax.set_xlim(YA - LT - 9, YB + 6)
ax.set_ylim(XC - A_OUT - 4, XC + A_OUT + 3.5)
ax.set_aspect("equal")
ax.set_xlabel("Y along the drain (mm)")
ax.set_ylabel("X (mm)")
ax.set_title("top view of the simulated fluid (light blue): sheath lumen 119.4 mm + 12 mm of outflow tube")
# ---- side view
ax = fig.add_subplot(gs[1, 0])
ax.add_patch(Rectangle((YA, ZC - B_OUT), YB - YA + 0.6, 2 * B_OUT, fc=WALL, ec="k", lw=0.8))
ax.add_patch(Rectangle((YA, ZC - B_OUT + 0.6), YB - YA, 2 * B_OUT - 1.2, fc=FL, ec="none"))
ax.add_patch(Rectangle((YA - LT, ZC - R_TUBE), LT, 2 * R_TUBE, fc=FL, ec="k", lw=0.8))
ax.add_patch(Rectangle((YB, ZC - R_NOZ), LN + 0.6, 2 * R_NOZ, fc=IRR, ec="k", lw=0.6))
ax.annotate("", xy=(YB - 14, ZC), xytext=(YB - 1, ZC), arrowprops=dict(arrowstyle="->", color=IRR, lw=2.5))
ax.annotate("", xy=(YA - LT - 5, ZC), xytext=(YA - LT, ZC), arrowprops=dict(arrowstyle="->", color="k", lw=2.5))
for name, (hs, c) in GROUPS.items():
    for y in sorted({h[1] for h in hs}):
        ax.add_patch(Rectangle((y - 0.5, ZC + B_OUT - 2.2), 1.0, 2.2, fc=c, ec="k", lw=0.5))
        ax.annotate("", xy=(y, ZC + B_OUT - 0.2), xytext=(y, ZC + B_OUT + 2.6), arrowprops=dict(arrowstyle="<->", color=c, lw=1.2))
ax.text(0, ZC + B_OUT + 3.1, "open holes: free to flow in or out, outside pressure 0 (drain lying in fluid)", ha="center", fontsize=10)
ax.text(30, ZC - B_OUT - 2.3, "all other walls closed (blocked holes = wall)", ha="center", fontsize=10)
ax.set_xlim(YA - LT - 9, YB + 6)
ax.set_ylim(ZC - B_OUT - 3.5, ZC + B_OUT + 4.2)
ax.set_aspect("equal")
ax.set_xlabel("Y along the drain (mm)")
ax.set_ylabel("Z (mm)")
ax.set_title("side view (rib profile not drawn here; see the cross-section)")
# ---- cross-section
ax = fig.add_subplot(gs[:, 1])
ax.add_patch(Polygon(skin, closed=True, fc=WALL, ec="k", lw=0.8))
ax.add_patch(Polygon(lumen, closed=True, fc=FL, ec="k", lw=0.8))
ax.add_patch(Circle((XC, ZC), R_TUBE, fc="none", ec="k", ls="--", lw=1))
ax.add_patch(Circle((XC, ZC), R_NOZ, fc=IRR, ec="k", lw=0.6))
ax.text(XC, ZC - 1.3, "irrigation outlet", ha="center", fontsize=9, color=IRR)
ax.text(XC, ZC + R_TUBE + 0.25, "outflow tube (dashed)", ha="center", fontsize=9)
for x in (0.0, 3.3):
    zs = ZC + B_OUT * np.sqrt(1 - ((x - XC) / A_OUT) ** 2)
    ax.add_patch(Rectangle((x - 0.5, zs - 2.4), 1.0, 2.4, fc="#d62728", ec="k", lw=0.5, alpha=0.8))
ax.text(1.6, 0.6, "open holes d = 1.0 mm", ha="center", fontsize=9, color="#d62728")
ax.set_xlim(XC - A_OUT - 0.8, XC + A_OUT + 0.8)
ax.set_ylim(ZC - B_OUT - 0.8, ZC + B_OUT + 1.6)
ax.set_aspect("equal")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Z (mm)")
ax.set_title("cross-section used in the model:\nreal USZ sheath with its ribs, internal tube removed\n(fluid area %.0f mm2)" % area(lumen))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "setup.png"), dpi=110)

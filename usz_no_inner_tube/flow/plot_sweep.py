"""Graph of the outlet-position sweep: old fluid left in the lumen against the outlet distance from the tip.

  python plot_sweep.py   ->  results/sweep.png, results/sweep.json   (reads results/*/flush.json)
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.load(open(f)) for f in glob.glob(os.path.join(HERE, "results", "*", "flush.json"))]
rows = [r for r in rows if r["T"] >= 600]
d1 = sorted([r for r in rows if abs(r["d_mm"] - 1.0) < 1e-6], key=lambda r: r["s_mm"])
d2 = sorted([r for r in rows if abs(r["d_mm"] - 2.0) < 1e-6], key=lambda r: r["s_mm"])
C1, C2, C3 = "#1f6feb", "#d1242f", "#6e7781"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.1))

a = axs[0]
s = np.array([r["s_mm"] for r in d1])
for key, col in (("300s", C1), ("600s", C2)):
    a.plot(s, [100 * r["old_left"][key] for r in d1], "o-", color=col, ms=4, label=f"1 mm outlet, after {int(key[:-1]) // 60} min")
    if d2:
        a.plot([r["s_mm"] for r in d2], [100 * r["old_left"][key] for r in d2], "s", color=col, mfc="none", ms=7,
               label=f"2 mm outlet, after {int(key[:-1]) // 60} min")
a.set_xlabel("outlet distance from the tip (mm)"); a.set_ylabel("old fluid left in the lumen (%)")
a.set_ylim(0, 100); a.set_xlim(-4, 104); a.legend(frameon=False, fontsize=7, loc="lower right")
a.set_title("A  Whole lumen", fontsize=9, loc="left")

a = axs[1]
k = s > 0
a.plot(s[k], [100 * r["old_left_behind_outlet"]["600s"] for r in np.array(d1)[k]], "o-", color=C2, ms=4, label="between tip and outlet")
a.plot(s, [100 * r["old_left_ahead_of_outlet"]["600s"] for r in d1], "o-", color=C1, ms=4, label="between outlet and suction")
a.set_xlabel("outlet distance from the tip (mm)"); a.set_ylabel("old fluid left after 10 min (%)")
a.set_ylim(0, 100); a.set_xlim(-4, 104); a.legend(frameon=False, fontsize=7, loc="center right")
a.set_title("B  The two parts of the lumen", fontsize=9, loc="left")

a = axs[2]
cm = plt.get_cmap("viridis")
for r in d1:
    a.plot(r["curve"]["t"], 100 * np.array(r["curve"]["left"]), color=cm(r["s_mm"] / 110), label=f"{r['s_mm']:.0f} mm")
for r in d2:
    a.plot(r["curve"]["t"], 100 * np.array(r["curve"]["left"]), color=cm(r["s_mm"] / 110), ls="--", label=f"{r['s_mm']:.0f} mm, 2 mm outlet")
a.set_xlabel("time (s)"); a.set_ylabel("old fluid left in the lumen (%)"); a.set_ylim(0, 100); a.set_xlim(0, 600)
a.legend(frameon=False, fontsize=6.5, ncol=2, loc="lower left", title="outlet position", title_fontsize=7)
a.set_title("C  Wash-out curves", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "results", "sweep.png"), dpi=200); plt.close(fig)
json.dump([{k: v for k, v in r.items() if k != "curve"} for r in d1 + d2], open(os.path.join(HERE, "results", "sweep.json"), "w"), indent=1)
for r in d1 + d2:
    print(f"d={r['d_mm']:.0f} s={r['s_mm']:5.0f}  left 5 min {100*r['old_left']['300s']:5.1f} %  10 min {100*r['old_left']['600s']:5.1f} %")

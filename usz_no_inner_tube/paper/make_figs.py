"""Figures for jet_loop_maths.tex: analytical jet model against the blocked-sheath simulation."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "..", "flow", "results", "blocked", "jet_check.json")))
x = np.array([r["x"] for r in rows]); uc = np.array([r["uc"] for r in rows])
qb = np.array([r["q_back"] for r in rows]); qf = np.array([r["q_forward"] for r in rows])
rh = np.array([r["r_half"] for r in rows])

# ---- parameters (SI)
nu, rho = 1e-6, 1000.0
Q = 250e-6 / 3600.0
d = 1e-3
A = 93.3e-6
Uj = Q / (np.pi * d ** 2 / 4)
Re = Uj * d / nu
K = 4.0 / 3.0 * Uj ** 2 * np.pi * d ** 2 / 4
x0 = Re * d / 16
Req = np.sqrt(A / np.pi)
c_half = 2 * np.sqrt(np.sqrt(2) - 1) * np.sqrt(16 * np.pi / 3)      # r_half = c_half nu x / sqrt(K)
spread = c_half * nu / np.sqrt(K)
x_end = Req / (2.5 * spread) - x0
ML = 60e6                                                            # m3/s -> mL/min
print(f"Uj={Uj*1e3:.1f} mm/s Re={Re:.1f} K={K:.3e} x0={x0*1e3:.2f} mm Req={Req*1e3:.2f} mm c_half={c_half:.3f} "
      f"spread={spread:.4f} x_end={x_end*1e3:.1f} mm 8pi nu={8*np.pi*nu*ML*1e-3:.3f} mL/min per mm "
      f"Ct={Q/A/np.sqrt(K/A):.3f} dp={rho*K/A:.3f} Pa")
C1, C2, C3 = "#1f6feb", "#d1242f", "#6e7781"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- figure 1: sketch
fig, ax = plt.subplots(figsize=(7.2, 2.5))
L, H = 120.0, 16.0
ax.add_patch(Rectangle((0, -H / 2), L, H, fc="#eef3f9", ec="#333", lw=1.2))
ax.add_patch(Rectangle((-5, -0.8), 5, 1.6, fc="#35a7ff", ec="#333", lw=0.8))
xe = 38.0
ax.add_patch(Polygon([[0, -0.6], [0, 0.6], [xe, 6.5], [xe, -6.5]], fc="#35a7ff", alpha=0.35, ec="none"))
ax.annotate("", xy=(30, 0), xytext=(1, 0), arrowprops=dict(arrowstyle="-|>", color=C1, lw=2))
for s in (1, -1):
    ax.add_patch(FancyArrowPatch((31, s * 2.0), (8, s * 6.6), connectionstyle=f"arc3,rad={0.55 * s}",
                                 arrowstyle="-|>", mutation_scale=10, color=C2, lw=1.4))
    ax.add_patch(FancyArrowPatch((9, s * 6.4), (5, s * 1.2), connectionstyle=f"arc3,rad={0.5 * s}",
                                 arrowstyle="-|>", mutation_scale=8, color=C2, lw=1.0))
for yy in (-5, -2.5, 0, 2.5, 5):
    ln = 16 * (1 - (yy / 8) ** 2)
    ax.annotate("", xy=(70 + ln, yy), xytext=(70, yy), arrowprops=dict(arrowstyle="-|>", color=C3, lw=1))
ax.plot([xe, xe], [-H / 2, H / 2], ls="--", color="#333", lw=0.8)
ax.text(xe, H / 2 + 1, r"$x_{\rm end}$", ha="center")
ax.text(15, 1.3, "jet", color=C1)
ax.text(12, -4.9, r"return flow $Q_{\rm back}(x)$", color=C2, fontsize=8)
ax.text(62, 9.3, r"slow duct flow, mean $U_m=Q/A$", color=C3, fontsize=8)
ax.text(-6, 2.2, r"$Q$, $d$", ha="center")
ax.annotate("", xy=(L + 9, 0), xytext=(L + 1, 0), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
ax.text(L + 2, 2, "suction", fontsize=8)
ax.annotate("", xy=(52, -H / 2 - 2.2), xytext=(0, -H / 2 - 2.2), arrowprops=dict(arrowstyle="-|>", color="#333", lw=0.8))
ax.text(54, -H / 2 - 3.0, r"$x$")
ax.text(84, -6.9, "all side holes blocked", fontsize=8, color="#333")
ax.set_xlim(-9, L + 12); ax.set_ylim(-H / 2 - 4, H / 2 + 4); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(os.path.join(HERE, "fig_sketch.png"), dpi=220, bbox_inches="tight"); plt.close(fig)

# ---- figure 2: jet model against simulation
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6))
xx = np.linspace(0.3, 60, 300) * 1e-3
a = axs[0]
a.loglog(x, uc, "o", ms=3, color=C1, label="simulation")
a.loglog(xx * 1e3, 3 * K / (8 * np.pi * nu * (xx + x0)) * 1e3, color=C2, label=r"free jet, eq. (5)")
a.axhline(2.22, color=C3, ls=":", lw=1); a.text(0.6, 2.5, "duct flow, 2.2 mm/s", color=C3, fontsize=7)
a.axvline(38, color="#333", ls="--", lw=0.7)
a.set_xlabel("distance from outlet x (mm)"); a.set_ylabel("centre-line speed (mm/s)"); a.legend(frameon=False, fontsize=7)
a = axs[1]
k = x <= 40
a.plot(x[k], rh[k], "o", ms=3, color=C1, label="simulation")
a.plot(xx * 1e3, spread * (xx + x0) * 1e3, color=C2, label=r"free jet, eq. (6)")
a.set_xlim(0, 42); a.set_ylim(0, 3.2)
a.set_xlabel("x (mm)"); a.set_ylabel("jet half-width (mm)"); a.legend(frameon=False, fontsize=7)
a = axs[2]
a.plot(x[k], qb[k], "o", ms=3, color=C1, label="simulation")
a.plot(xx * 1e3, 8 * np.pi * nu * xx * ML, color=C2, label=r"free jet $8\pi\nu x$")
xs_ = np.linspace(0, 38, 200)
a.plot(xs_, 8 * np.pi * nu * xs_ * 1e-3 * ML * (1 - xs_ / 38) ** 1.5, color=C3, ls="--", label="closure, eq. (12)")
a.axhline(Q * ML, color="#333", ls=":", lw=0.8); a.text(27, Q * ML + 0.4, "irrigation Q", fontsize=7)
a.set_xlim(0, 42); a.set_ylim(0, 16)
a.set_xlabel("x (mm)"); a.set_ylabel("return flow (mL/min)"); a.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_jet.png"), dpi=220); plt.close(fig)

# ---- figure 3: washout and design scaling
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
a = axs[0]
t = np.linspace(0, 600, 600)
V_loop = A * 38e-3
tau = V_loop / Q
t_res = (120e-3 - 38e-3) / (Q / A)
a.plot(t, np.exp(-t / tau), color=C1, label=fr"tip zone (stirred), $\tau$={tau:.0f} s")
phi = np.where(t > t_res / 2, t_res / (4 * np.maximum(t, 1e-9)), 1 - t / t_res)
a.plot(t, phi, color=C2, label=fr"duct section, parabolic profile, $t_r$={t_res:.0f} s")
a.plot(t, np.clip(1 - t / t_res, 0, 1), color=C3, ls="--", label="duct section, ideal plug")
a.set_xlabel("time (s)"); a.set_ylabel("fraction of old fluid left"); a.set_ylim(0, 1.02); a.set_xlim(0, 600)
a.legend(frameon=False, fontsize=7)
a = axs[1]
dd = np.linspace(0.6, 2.6, 100) * 1e-3
for qh, col in ((125, C3), (250, C1), (500, C2)):
    q = qh * 1e-6 / 3600
    re = q / (np.pi * dd ** 2 / 4) * dd / nu
    xe_ = re * (0.0776 * Req - dd / 16)
    a.plot(dd * 1e3, np.clip(xe_, 0, None) * 1e3, color=col, label=f"{qh} mL/h")
a.plot([1.0], [38], "k*", ms=9, label="simulation")
a.axhline(120, color="#333", ls=":", lw=0.8); a.text(1.75, 123, "sheath length", fontsize=7)
a.set_ylim(0, 140); a.set_xlabel("outlet diameter d (mm)"); a.set_ylabel(r"loop length $x_{\rm end}$ (mm)")
a.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_wash.png"), dpi=220); plt.close(fig)
print(f"tau={tau:.1f} s  t_res={t_res:.1f} s  V_loop={V_loop*1e6:.2f} mL  turnover={V_loop*ML/10.1*60:.0f} s")

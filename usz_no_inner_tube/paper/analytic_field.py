"""Analytical flow field of the blocked sheath (jet + return flow in a round tube of the same area) drawn against the simulation.

Model, stream function psi(x, r) with u = (1/r) d(psi)/dr, s = r^2/R^2, X = x + x0:
    psi = m * nu X F(xi)  +  (1 - m) * Q/2pi * (2s - s^2)  -  B(x) * (3s^2 - 2s^3)
    F = xi^2/(1 + xi^2/4),  xi = sqrt(3K/16pi) r/(nu X)          (Schlichting jet)
    B = m * (nu X F(xi_R) - Q/2pi)                               (net flow through every section = Q)
  A: m = 1 for x < x_end(eq. 10), no fitted constant
  B: m falls smoothly from 1 at 0.5 x_end to 0 at 1.5 x_end (cosine), no fitted constant
Writes fig_field.png and fig_profiles.png.
"""
import os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "flow", "runs", "blocked")
XC, ZC, YB = 0.78, -4.82, 59.9
nu = 1e-6
Q = 250e-6 / 3600
d = 1e-3
A = 93.3e-6
R = np.sqrt(A / np.pi)
Uj = Q / (np.pi * d ** 2 / 4)
K = 4 / 3 * Uj ** 2 * np.pi * d ** 2 / 4
x0 = Uj * d / nu * d / 16
a = np.sqrt(3 * K / (16 * np.pi))
x_end_model = R * np.sqrt(K) / (13.2 * nu) - x0
ML = 60e6


def F(xi):
    return xi ** 2 / (1 + xi ** 2 / 4)


def field(x, r, closure, x_end):
    """axial speed u(x, r) in m/s and return flow Q_back(x) in m3/s; x, r 1-D arrays in m"""
    Xg, Rg = np.meshgrid(x, r, indexing="ij")
    X = Xg + x0
    t = np.clip((Xg / x_end - 0.5), 0, 1)                        # jet handed over to duct flow between 0.5 and 1.5 x_end
    m = 0.5 * (1 + np.cos(np.pi * t)) if closure else (Xg < x_end).astype(float)
    s = (Rg / R) ** 2
    B = m * (nu * X * F(a * R / (nu * X)) - Q / (2 * np.pi))
    psi = m * nu * X * F(a * Rg / (nu * X)) + (1 - m) * Q / (2 * np.pi) * (2 * s - s ** 2) - B * (3 * s ** 2 - 2 * s ** 3)
    u = np.gradient(psi, r, axis=1) / np.maximum(Rg, 1e-9)
    u[:, 0] = u[:, 1]
    dA = 2 * np.pi * Rg * np.gradient(r)[None, :]
    qb = (np.clip(-u, 0, None) * dA).sum(1)
    return psi, u, qb


x = np.linspace(0.2e-3, 60e-3, 400)
r = np.linspace(0, R, 300)
psiA, uA, qbA = field(x, r, False, x_end_model)
psiB, uB, qbB = field(x, r, True, x_end_model)

# ---- simulation on the two mid-planes
m_ = np.load(os.path.join(RUN, "mesh.npz")); res = np.load(os.path.join(RUN, "result.npz"))
tets = m_["tets"]
g = pv.UnstructuredGrid(np.hstack([np.full((len(tets), 1), 4), tets]).ravel(),
                        np.full(len(tets), pv.CellType.TETRA), m_["xyz"])
g["v"] = res["v"] * 1e3
xs = np.linspace(0.2, 60, 240)


def plane(lat_axis, half):
    lat = np.linspace(-half, half, 121)
    Xs, Ls = np.meshgrid(xs, lat, indexing="ij")
    P = np.zeros((Xs.size, 3))
    P[:, 1] = YB - Xs.ravel()
    P[:, 0], P[:, 2] = XC, ZC
    P[:, lat_axis] += Ls.ravel()
    sm = pv.PolyData(P).sample(g)
    ok = np.asarray(sm["vtkValidPointMask"]).astype(bool).reshape(Xs.shape)
    v = np.asarray(sm["v"])
    u = np.where(ok, -v[:, 1].reshape(Xs.shape), np.nan)
    w = np.where(ok, v[:, lat_axis].reshape(Xs.shape), np.nan)
    return lat, u, w


latx, ux, wx = plane(0, 8.6)        # wide direction of the oval
latz, uz, wz = plane(2, 4.5)        # narrow direction

norm = SymLogNorm(linthresh=1.0, vmin=-180, vmax=180, base=10)
fig, axs = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[10.9, 17.2, 9.0], hspace=0.28))
rr = np.r_[-r[::-1], r] * 1e3


def draw_model(ax, psi, u, title):
    U = np.c_[u[:, ::-1], u].T * 1e3
    im = ax.pcolormesh(x * 1e3, rr, U, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
    P = np.c_[psi[:, ::-1], psi].T
    lev = np.unique(np.r_[np.linspace(P.min(), 0, 6)[:-1], Q / (2 * np.pi) * np.linspace(0.08, 0.96, 6),
                          np.linspace(Q / (2 * np.pi), P.max(), 7)[1:]])
    ax.contour(x * 1e3, rr, P, levels=lev, colors="k", linewidths=0.5, linestyles="solid")
    ax.set_title(title, fontsize=9, loc="left"); ax.set_ylabel("r (mm)")
    return im


draw_model(axs[0], psiB, uB, "A  Analytical model: jet + return flow in a round tube of the same area (no fitted constant)")
for ax, lat, u, w, ttl in ((axs[1], latx, ux, wx, "B  Simulation, cut along the wide direction of the oval lumen"),
                           (axs[2], latz, uz, wz, "C  Simulation, cut along the narrow direction of the oval lumen")):
    im = ax.pcolormesh(xs, lat, u.T, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
    ax.streamplot(xs, lat, np.nan_to_num(u.T), np.nan_to_num(w.T), color="k", linewidth=0.5, density=(2.2, 0.9),
                  arrowsize=0.5, broken_streamlines=False)
    ax.set_title(ttl, fontsize=9, loc="left"); ax.set_ylabel("across (mm)")
    ax.set_ylim(lat[0], lat[-1])
axs[2].set_xlabel("distance from the irrigation outlet x (mm)")
cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.7, pad=0.1, aspect=40, ticks=[-100, -10, -1, 0, 1, 10, 100])
cb.ax.set_xticklabels(["-100", "-10", "-1", "0", "1", "10", "100"])
cb.set_label("axial speed (mm/s): red towards suction, blue back towards the tip")
fig.savefig(os.path.join(HERE, "fig_field.png"), dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- profiles and return flow
import json
rows = json.load(open(os.path.join(HERE, "..", "flow", "results", "blocked", "jet_check.json")))
fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.5))
for ax, xq in zip(axs[:3], (5, 15, 30)):
    i = np.argmin(abs(x * 1e3 - xq)); j = np.argmin(abs(xs - xq))
    ax.plot(rr, np.r_[uB[i, ::-1], uB[i]] * 1e3, color="#d1242f", label="analytical")
    ax.plot(latx, ux[j], color="#1f6feb", lw=1, label="simulation, wide")
    ax.plot(latz, uz[j], color="#1f6feb", lw=1, ls=":", label="simulation, narrow")
    ax.axhline(0, color="k", lw=0.4); ax.set_xlim(-8.6, 8.6)
    ax.set_title(f"x = {xq} mm", fontsize=9); ax.set_xlabel("across (mm)")
axs[0].set_ylabel("axial speed (mm/s)"); axs[0].legend(frameon=False, fontsize=6)
ax = axs[3]
ax.plot([r_["x"] for r_ in rows], [r_["q_back"] for r_ in rows], "o", ms=2.5, color="#1f6feb", label="simulation")
ax.plot(x * 1e3, qbB * ML, color="#d1242f", label="analytical")
ax.set_xlim(0, 45); ax.set_xlabel("x (mm)"); ax.set_title("return flow (mL/min)", fontsize=9); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_profiles.png"), dpi=200); plt.close(fig)
for xq in (5, 10, 15, 20, 30):
    i = np.argmin(abs(x * 1e3 - xq))
    print(f"x={xq:2d}  Q_back  A={qbA[i]*ML:6.1f}  B={qbB[i]*ML:6.1f} mL/min   u_axis A={uA[i,0]*1e3:6.1f} B={uB[i,0]*1e3:6.1f} mm/s")
print("peak A", round(qbA.max() * ML, 1), "at", round(x[qbA.argmax()] * 1e3, 1), "| peak B", round(qbB.max() * ML, 1), "at", round(x[qbB.argmax()] * 1e3, 1))

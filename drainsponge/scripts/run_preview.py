"""
Run the reduced-order (axisymmetric) coupled model and make the result pictures.

  out/preview/preview.json       time histories + snapshots (used by the HTML viewer,
                                 the one-way FEBio model and the sealed OpenFOAM case)
  out/figures/05_preview_timeline.png
  out/figures/06_preview_fields.png
  out/figures/07_parameter_map.png   (with --sweep)

python scripts/run_preview.py [--config cfg.json] [--sweep]
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.preview import Preview

SNAPS = [0.0005, 0.002, 0.004, 0.008, 0.05, 5.0, 60.0, 600.0, 3600.0]


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return np.round(o.astype(float), 9).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def timeline_figure(h, pv, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array(h["t"])
    fig, axs = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    ax = axs[0, 0]
    ax.loglog(t, np.abs(h["Q_hole"]) * 0.06, "k", lw=2, label="total into the hole")
    ax.loglog(t, np.clip(h["Q_edge"], 1e-12, None) * 0.06, color="#4a90d9", lw=1.6, label="free liquid through the film")
    seep = np.array(h["Q_body"]) + np.array(h["Q_seep"])
    seep = np.where(seep > 0, seep, np.nan)
    ax.loglog(t, seep * 0.06, color="#c0392b", lw=1.6, label="liquid out of the sponge (Darcy)")
    ax.set_ylim(1e-8, 10)
    ax.set_ylabel("flow (mL/min, one hole)"); ax.legend(fontsize=9); ax.set_title("where the liquid comes from")
    ax.grid(True, which="both", alpha=0.3)
    ax = axs[0, 1]
    ax.semilogx(t, np.array(h["h_rim"]) * 1e3, color="#4a90d9", lw=2, label="film at the hole rim")
    ax.semilogx(t, np.array(h["intrusion"]) * 1e3, color="#c0392b", lw=2, label="tissue depth into the hole")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("um"); ax.legend(fontsize=9); ax.set_title("tissue pulled onto / into the hole")
    ax.grid(True, which="both", alpha=0.3)
    ax = axs[1, 0]
    ax.semilogx(t, np.array(h["p_hole"]) * 1e-3, "k", lw=2)
    ax.set_ylabel("kPa"); ax.set_xlabel("time after suction on (s)"); ax.set_title("pressure in the hole")
    ax.grid(True, which="both", alpha=0.3)
    ax = axs[1, 1]
    ax.semilogx(t, np.array(h["V_cum_seep"]) * 1e3, color="#c0392b", lw=2, label="drawn out of the sponge")
    ax.set_ylabel("cumulative volume (nL = 1e-3 mm^3)"); ax.set_xlabel("time after suction on (s)")
    ax.set_title("liquid pulled out of the tissue"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    ts = h.get("t_seal")
    for a in axs.ravel():
        if ts:
            a.axvline(ts, color="#888", ls="--", lw=1)
    if ts:
        axs[0, 1].annotate(f"sealed at {ts*1e3:.1f} ms", xy=(ts, 30), xytext=(ts * 3, 38), fontsize=9,
                           arrowprops=dict(arrowstyle="->", color="#666"))
    fig.suptitle("Single hole + sponge tissue (reduced-order axisymmetric model)  E=%.0f kPa, k=%.0e m4/Ns, "
                 "suction %.0f kPa, gap %.0f um" % (pv.E * 1e3, pv.k * 1e-12, pv.dp * 1e3, pv.g0 * 1e3), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fields_figure(snaps, pv, path, pick=(1, 3, 5, 8)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    fig, axs = plt.subplots(len(pick), 1, figsize=(9.2, 3.0 * len(pick)), constrained_layout=True)
    a, t, g0, R = pv.a, pv.t_wall, pv.g0, pv.R
    for ax, i in zip(axs, pick):
        s = snaps[i]
        r, z, u, p = s["r2"], s["z2"], s["u"], s["p_Pa"]
        Rg, Zg = np.meshgrid(r, z)
        Xd, Zd = Rg + u[..., 0], g0 + Zg + u[..., 1]           # true scale, real position above the skin
        Xf = np.hstack([-Xd[:, ::-1], Xd]); Zf = np.hstack([Zd[:, ::-1], Zd]); Pf = np.hstack([p[:, ::-1], p])
        lim = max(abs(p).max(), 1)
        cs = ax.pcolormesh(Xf, Zf, Pf * 1e-3, cmap="RdBu", vmin=-lim * 1e-3, vmax=lim * 1e-3, shading="gouraud")
        # liquid flow inside the sponge: streamlines of the Darcy flux (undeformed positions)
        xc, wv = s["xc"], s["w_mm_s"]
        gx = np.linspace(-R * 0.98, R * 0.98, 160); gz = np.linspace(0.01, 1.0, 70)
        GX, GZ = np.meshgrid(gx, gz)
        pts = np.r_[xc, xc * [-1, 1]]; W = np.r_[wv, wv * [-1, 1]]
        U = griddata(pts, W[:, 0], (GX, GZ)); V = griddata(pts, W[:, 1], (GX, GZ))
        mag = np.hypot(U, V)
        lw = 0.4 + 1.6 * np.clip(np.log10(mag / np.nanmax(mag) + 1e-12) + 3, 0, 3) / 3
        ax.streamplot(GX, GZ + g0, U, V, color="k", linewidth=lw, density=1.3, arrowsize=0.8)
        # drain wall, hole, film
        ax.fill_between([-R, -a], [-t, -t], [0, 0], color="#8a9bb0", zorder=3)
        ax.fill_between([a, R], [-t, -t], [0, 0], color="#8a9bb0", zorder=3)
        hf, ub = s["film_h"], u[0, :, 1]
        o = r >= a
        for sg in (1, -1):
            ax.fill_between(sg * r[o], 0, hf[o], color="#4a90d9", alpha=0.85, lw=0, zorder=4)
        ax.set_xlim(-R, R); ax.set_ylim(-t, 0.75); ax.set_aspect("equal")
        sealed = "sealed on the rim" if hf[o][:3].max() < 1.5 * pv.hmin else "film open"
        ax.set_title(f"t = {s['t']:.3g} s   hole {s['p_hole_Pa']*1e-3:.2f} kPa   film at rim "
                     f"{hf[o][0]*1e3:.1f} um ({sealed})   depth into hole {max(-(g0+ub[~o].min()),0)*1e3:.0f} um",
                     fontsize=10)
        fig.colorbar(cs, ax=ax, label="pore pressure (kPa)", fraction=0.02, pad=0.01)
    axs[-1].set_xlabel("mm across the hole (true scale; blue = liquid film, lines = liquid flowing inside the sponge)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summarize(h):
    t = np.array(h["t"])
    seal = [i for i, s in enumerate(h["sealed"]) if s]
    h["t_seal"] = float(t[seal[0]]) if seal else None
    seep = (np.array(h["Q_body"]) + np.array(h["Q_seep"]))
    dt = np.diff(np.concatenate([[0.0], t]))
    h["V_cum_seep"] = np.cumsum(seep * dt).tolist()
    return dict(t_seal_s=h["t_seal"], peak_flow_mL_min=float(np.max(np.abs(h["Q_hole"])) * 0.06),
                flow_1h_mL_min=float(h["Q_hole"][-1] * 0.06),
                tissue_flow_1h_mL_min=float(seep[-1] * 0.06),
                tissue_volume_1h_mm3=float(h["V_cum_seep"][-1]),
                intrusion_1h_um=float(h["intrusion"][-1] * 1e3),
                film_at_rim_1h_um=float(h["h_rim"][-1] * 1e3),
                max_mass_error=float(np.max(np.abs(h["mass_err"][1:]))))


def sweep(cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Es = [5e3, 10e3, 20e3, 50e3]
    dps = [2e3, 5e3, 10e3, 20e3]
    ks = [1e-15, 1e-14, 1e-13, 1e-12]
    I = np.zeros((len(Es), len(dps))); TS = np.zeros_like(I)
    for i, E in enumerate(Es):
        for j, dp in enumerate(dps):
            c = config.load(tissue={"E_Pa": E}, suction={"dp_Pa": dp})
            pv = Preview(c); h, _ = pv.run(t_end=3600, n_steps=60, verbose=False)
            sm = summarize(h)
            I[i, j] = sm["intrusion_1h_um"]; TS[i, j] = sm["t_seal_s"] or np.nan
            print(f"E={E:.0f} dp={dp:.0f}: intrusion {I[i,j]:.0f} um, seal {TS[i,j]:.3g} s")
    Qk = []
    for k in ks:
        c = config.load(tissue={"perm_m4_Ns": k})
        pv = Preview(c); h, _ = pv.run(t_end=3600, n_steps=60, verbose=False)
        sm = summarize(h)
        Qk.append((sm["tissue_flow_1h_mL_min"], sm["tissue_volume_1h_mm3"], sm["flow_1h_mL_min"]))
        print(f"k={k:.0e}: tissue flow {Qk[-1][0]:.3g} mL/min, total {Qk[-1][2]:.3g}")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axs[0]
    im = ax.imshow(I, origin="lower", cmap="magma_r", aspect="auto")
    for i in range(len(Es)):
        for j in range(len(dps)):
            frac = I[i, j] / 600
            txt = (f"{I[i,j]:.0f} um\n({frac*100:.0f}% wall)" + ("\nOCCLUSION" if frac > 0.85 else "")) \
                if I[i, j] > 0 else f"film stays open\n(tissue {-I[i,j]:.0f} um\nabove the rim)"
            ax.text(j, i, txt,
                    ha="center", va="center", fontsize=9, color="w" if I[i, j] > 0.5 * I.max() else "k")
    ax.set_xticks(range(len(dps))); ax.set_xticklabels([f"{d/1e3:.0f}" for d in dps])
    ax.set_yticks(range(len(Es))); ax.set_yticklabels([f"{e/1e3:.0f}" for e in Es])
    ax.set_xlabel("suction (kPa)"); ax.set_ylabel("tissue modulus E (kPa)")
    ax.set_title("tissue depth into the 1 mm hole after 1 h\n(linear model; values > ~200 um are only indicative)")
    fig.colorbar(im, ax=ax, label="um")
    ax = axs[1]
    Qk = np.array(Qk)
    ax.loglog(ks, Qk[:, 0], "o-", color="#c0392b", lw=2, label="liquid out of the sponge")
    ax.loglog(ks, Qk[:, 2], "s-", color="k", lw=2, label="total into the hole (incl. leak under the seal)")
    ax.set_xlabel("tissue permeability k (m^4/(N s))"); ax.set_ylabel("flow after 1 h (mL/min, one hole)")
    ax.set_title("how much liquid the tissue itself gives (E=20 kPa, 5 kPa)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "07_parameter_map.png"), dpi=140)
    plt.close(fig)
    return dict(E_Pa=Es, dp_Pa=dps, intrusion_um=I.tolist(), t_seal_s=TS.tolist(),
                k=ks, tissue_flow_mL_min=Qk[:, 0].tolist(), total_flow_mL_min=Qk[:, 2].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--figs-only", action="store_true", help="redraw figures from out/preview/preview.json")
    a = ap.parse_args()
    cfg = config.load(a.config)
    figs = os.path.join(a.out, "figures"); os.makedirs(figs, exist_ok=True)
    pdir = os.path.join(a.out, "preview"); os.makedirs(pdir, exist_ok=True)
    t0 = time.time()
    pv = Preview(cfg)
    if a.figs_only:
        res = json.load(open(os.path.join(pdir, "preview.json")))
        sn = [{k: (np.array(v) if isinstance(v, list) else v) for k, v in s.items()} for s in res["snapshots"]]
        timeline_figure(res["history"], pv, os.path.join(figs, "05_preview_timeline.png"))
        fields_figure(sn, pv, os.path.join(figs, "06_preview_fields.png"))
        return
    h, snaps = pv.run(t_end=cfg["time"]["t_end_s"], n_steps=a.steps, snap_times=SNAPS)
    sm = summarize(h)
    print(json.dumps(sm, indent=1), f"({time.time()-t0:.0f} s)")
    timeline_figure(h, pv, os.path.join(figs, "05_preview_timeline.png"))
    fields_figure(snaps, pv, os.path.join(figs, "06_preview_fields.png"))
    res = dict(summary=sm, config=cfg, history=h, snapshots=snaps,
               geometry=dict(a=pv.a, R=pv.R, H=pv.H, g0=pv.g0, wall=pv.t_wall))
    if a.sweep:
        res["sweep"] = sweep(cfg, figs)
    json.dump(to_jsonable(res), open(os.path.join(pdir, "preview.json"), "w"))


if __name__ == "__main__":
    main()

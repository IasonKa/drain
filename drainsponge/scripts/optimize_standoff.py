"""
Design screening of the standoff ring with the reduced-order (preview) model.

Design variables (around the new offset design: h 0.40, r_in 0.55, width 0.30, slot 0.35, 4 slots):
  standoff_h   ring height above the skin (mm)
  ring_w       ring width r_out - r_in (mm), r_in fixed at 0.55 (just outside the Ø1 hole)
  slot_w       slot width (mm)
Each design is evaluated in two scenarios:
  worst  : no free liquid, soft tissue E = 5 kPa, suction 10 kPa  -> does tissue reach the hole?
  pool   : free liquid at the window edge, E = 10 kPa, 10 kPa     -> does the hole stay open?
Objectives: clearance (tissue above the hole rim, worst case, maximise), ring contact
pressure (tissue loading, minimise), free-liquid flow (maximise), ring height (minimise).
Results: out/optim/results.json, out/figures/08_standoff_screening.png

  python scripts/optimize_standoff.py                  # all designs locally (process pool)
  python scripts/optimize_standoff.py --index $SLURM_ARRAY_TASK_ID   # one design (cluster array)
  python scripts/optimize_standoff.py --collect        # merge + figure
The preview is axisymmetric: tissue sagging into the slots is NOT captured -> confirm the
chosen design with the 3D coupled model (hpc/run_coupled.sbatch).
"""
import argparse, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config
from drainsponge.preview import Preview

H = [0.2, 0.3, 0.4, 0.5, 0.6]
WIDTH = [0.2, 0.3, 0.45]
SLOT = [0.2, 0.35, 0.5]
SCEN = {"worst": dict(tissue={"E_Pa": 5000.0}, suction={"dp_Pa": 10000.0}, boundary={"film_edge": "closed"}),
        "severe": dict(tissue={"E_Pa": 10000.0}, suction={"dp_Pa": 10000.0}, boundary={"film_edge": "closed"}),
        "pool": dict(tissue={"E_Pa": 10000.0}, suction={"dp_Pa": 10000.0}, boundary={"film_edge": "open"})}


def designs():
    out = [dict(name="current", standoff_h=0.0)]
    for h, wd, s in itertools.product(H, WIDTH, SLOT):
        out.append(dict(name=f"h{h:.2f}_w{wd:.2f}_s{s:.2f}", standoff_h=h, standoff_rin=0.55,
                        standoff_rout=0.55 + wd, slot_w=s, n_slots=4))
    return out


def evaluate(d, steps=50, done=None):
    res = dict(done or d)
    for sc, over in SCEN.items():
        if sc in res:
            continue
        win = {k: v for k, v in d.items() if k != "name"}
        cfg = config.load(design="standoff" if d["standoff_h"] > 0 else "current", window=win, **over)
        pv = Preview(cfg)
        h, _ = pv.run(t_end=3600, n_steps=steps, verbose=False)
        ring_area = np.pi * (pv.rout ** 2 - pv.rin ** 2) * (1 - min(1, pv.n_slots * pv.slot_w / (np.pi * (pv.rin + pv.rout)))) \
            if pv.hs > 0 else np.nan
        res[sc] = dict(clearance_um=float(h["clearance"][-1] * 1e3),
                       Q_1h_mL_min=float(h["Q_hole"][-1] * 0.06),
                       ring_contact_kPa=float(h["ring_load"][-1] / ring_area * 1e3) if pv.hs > 0 else float("nan"),
                       mass_err=float(np.max(np.abs(h["mass_err"][1:]))))
    return res


def _eval(args):
    return evaluate(args[0], done=args[1])


def collect(folder, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [json.load(open(os.path.join(folder, f))) for f in sorted(os.listdir(folder)) if f.startswith("d_")]
    json.dump(R, open(os.path.join(folder, "results.json"), "w"), indent=1)
    cur = [r for r in R if r["standoff_h"] == 0]
    so = [r for r in R if r["standoff_h"] > 0]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    ax = axs[0]
    for sc, ls, lab in (("severe", "-", "E = 10 kPa"), ("worst", "--", "E = 5 kPa")):
        for wd, col in zip(WIDTH, ["#1b9e77", "#d95f02", "#7570b3"]):
            pts = sorted([(r["standoff_h"], r[sc]["clearance_um"]) for r in so
                          if abs(r["standoff_rout"] - 0.55 - wd) < 1e-6 and abs(r["slot_w"] - 0.35) < 1e-6])
            ax.plot(*zip(*pts), marker="o", ls=ls, color=col, lw=1.3, label=f"{lab}, ring width {wd}")
        if cur:
            ax.plot([0], [cur[0][sc]["clearance_um"]], marker="X", ms=10, color="k", ls="none",
                    label=f"current design, {lab}")
    ax.axhline(0, color="#c0392b", lw=1.5)
    ax.text(0.02, 15, "above 0: tissue stays out of the hole", color="#c0392b", fontsize=9)
    ax.set_xlabel("ring height (mm)  [0 = current design]"); ax.set_ylabel("lowest tissue point vs. hole rim after 1 h (um)")
    ax.set_title("no free liquid, 10 kPa suction (slot 0.35)")
    ax.legend(fontsize=7, ncol=1)
    ax = axs[1]
    for wd, mk in zip(WIDTH, ["o", "s", "^"]):
        pts = sorted([(r["standoff_h"], r["worst"]["ring_contact_kPa"]) for r in so
                      if abs(r["standoff_rout"] - 0.55 - wd) < 1e-6 and abs(r["slot_w"] - 0.35) < 1e-6])
        ax.plot(*zip(*pts), marker=mk, color="k", lw=1, label=f"ring width {wd} mm")
    ax.set_xlabel("ring height (mm)"); ax.set_ylabel("mean contact pressure on the ring tops (kPa)")
    ax.set_title("tissue loading on the rings (E = 5 kPa, slot 0.35)")
    ax.legend(fontsize=8)
    ax = axs[2]
    for s, col in zip(SLOT, ["#1b9e77", "#d95f02", "#7570b3"]):
        pts = sorted([(r["standoff_h"], r["pool"]["Q_1h_mL_min"]) for r in so
                      if abs(r["standoff_rout"] - 0.85) < 1e-6 and abs(r["slot_w"] - s) < 1e-6])
        ax.plot(*zip(*pts), marker="o", color=col, label=f"slot {s} mm")
    if cur:
        ax.axhline(cur[0]["pool"]["Q_1h_mL_min"], color="k", ls="--", lw=1, label="current design")
    ax.set_yscale("log")
    ax.set_xlabel("ring height (mm)"); ax.set_ylabel("flow into the hole after 1 h (mL/min)")
    ax.set_title("free liquid available (E = 10 kPa, 10 kPa), ring width 0.30")
    ax.legend(fontsize=8)
    fig.suptitle("Standoff ring screening (reduced-order axisymmetric model; slots modelled as ducts, "
                 "tissue sag into the slots not included)", fontsize=11)
    fig.tight_layout()
    os.makedirs(figs, exist_ok=True)
    fig.savefig(os.path.join(figs, "08_standoff_screening.png"), dpi=140)
    plt.close(fig)
    # recommendation: lowest ring that keeps >= 100 um clearance in the severe case (E 10 kPa),
    # then the best worst-case clearance, then the lowest ring contact pressure
    ok = [r for r in so if r["severe"]["clearance_um"] >= 100]
    ok.sort(key=lambda r: (r["standoff_h"], -r["worst"]["clearance_um"], r["worst"]["ring_contact_kPa"]))
    rec = ok[0] if ok else max(so, key=lambda r: r["severe"]["clearance_um"])
    base = [r for r in so if abs(r["standoff_h"] - 0.4) < 1e-6 and abs(r["standoff_rout"] - 0.85) < 1e-6
            and abs(r["slot_w"] - 0.35) < 1e-6]
    summary = dict(recommended=rec, baseline_new_design=base[0] if base else None,
                   current=cur[0] if cur else None, n_designs=len(R))
    json.dump(summary, open(os.path.join(folder, "summary.json"), "w"), indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/optim")
    ap.add_argument("--figs", default="out/figures")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = designs()
    if a.collect:
        print(json.dumps(collect(a.out, a.figs), indent=1))
        return
    todo = [a.index] if a.index is not None else range(len(D))
    def have(i):
        f = os.path.join(a.out, f"d_{i:03d}.json")
        return json.load(open(f)) if os.path.exists(f) else None
    todo = [i for i in todo if not (have(i) and all(k in have(i) for k in SCEN))]
    t0 = time.time()
    if a.index is not None or a.workers <= 1:
        for i in todo:
            json.dump(evaluate(D[i], done=have(i)), open(os.path.join(a.out, f"d_{i:03d}.json"), "w"), indent=1)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            for i, r in zip(todo, ex.map(_eval, [(D[i], have(i)) for i in todo])):
                json.dump(r, open(os.path.join(a.out, f"d_{i:03d}.json"), "w"), indent=1)
                print(f"{i:3d} {r['name']:22s} worst clearance {r['worst']['clearance_um']:7.1f} um  "
                      f"pool Q {r['pool']['Q_1h_mL_min']:.3g} mL/min  ({time.time()-t0:.0f}s)", flush=True)
    if a.index is None:
        print(json.dumps(collect(a.out, a.figs), indent=1))


if __name__ == "__main__":
    main()

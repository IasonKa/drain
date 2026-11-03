"""Results of a drainstop VOF run: pool level, flows and stop level, next to the model's prediction.

    python vof/analyse_run.py <run dir> [--plot]    # -> <run>/analysis/summary.txt, summary.json (+ PNG)
    python vof/analyse_run.py <run dir> --timing    # cost per step so far and the projected wall time

Pool level h, in mm above the lowest point of the skin (the h of the model), is read two ways:
  h_vol    from the liquid volume in the pool (all liquid minus the drain's cellZone 'lumen'),
           through the pool's volume table V(h) in case.json. Integral, so waves and the draw-down
           next to the drain do not disturb it. This is the level to compare with the model.
  h_probe  liquid column on world-vertical probe lines in the pool (median over the wet lines).
           Local; the spread between lines shows the draw-down.
Stop level: the level at which the pool's drain rate, averaged over stop_window_s, first falls below
q_stop_ml_min (the test that ends the run), and the level at which a straight line through the
late part of the drain rate Q(h) reaches zero (usable before the run has ended).
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def _time_dirs(run, name):
    base = os.path.join(run, "postProcessing", name)
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if _num(os.path.basename(d)) is not None]
    return sorted(dirs, key=lambda p: float(os.path.basename(p)))


def read_dat(run, name):
    """Rows (time, columns...) of a function object's .dat output, restarts joined (later wins)."""
    rows = {}
    for d in _time_dirs(run, name):
        for f in sorted(glob.glob(os.path.join(d, "*.dat"))):
            for line in open(f):
                if line.startswith("#"):
                    continue
                p = [_num(x) for x in line.replace("(", " ").replace(")", " ").split()]
                if len(p) >= 2 and None not in p:
                    rows[p[0]] = p[1:]
    if not rows:
        return np.zeros(0), np.zeros((0, 1))
    t = np.array(sorted(rows))
    n = min(len(rows[x]) for x in t)
    return t, np.array([rows[x][:n] for x in t])


def read_probes(run, name, field):
    """Times, values (n_t, n_probes) and probe coordinates (m) from a probes function object."""
    rows, coords = {}, None
    for d in _time_dirs(run, name):
        f = os.path.join(d, field)
        if not os.path.isfile(f):
            continue
        c = []
        for line in open(f):
            if line.startswith("#"):
                m = re.match(r"#\s*Probe\s+\d+\s*\(([^)]*)\)", line)
                if m:
                    c.append([float(x) for x in m.group(1).split()])
                continue
            p = [_num(x) for x in line.split()]
            if len(p) >= 2 and None not in p:
                rows[p[0]] = p[1:]
        coords = np.array(c) if c else coords
    if not rows:
        return np.zeros(0), np.zeros((0, 0)), coords
    t = np.array(sorted(rows))
    n = min(len(rows[x]) for x in t)
    return t, np.array([rows[x][:n] for x in t]), coords


def moving_slope(t, v, w):
    """Least-squares slope of v(t) over the trailing window w, at every sample."""
    out = np.full(len(t), np.nan)
    j = 0
    for i in range(len(t)):
        while t[j] < t[i] - w:
            j += 1
        if i - j >= 4 and t[i] - t[j] > 0.5 * w:
            tt, vv = t[j:i + 1], v[j:i + 1]
            out[i] = np.polyfit(tt - tt.mean(), vv, 1)[0]
    return out


def load(run):
    case = json.load(open(os.path.join(run, "case.json")))
    r = dict(case=case)
    t, va = read_dat(run, "liquidVolume")
    tl, vl = read_dat(run, "lumenVolume")
    if len(t) == 0:
        raise SystemExit(f"no postProcessing/liquidVolume in {run} yet")
    vl = np.interp(t, tl, vl[:, 0]) if len(tl) else np.zeros(len(t))
    Vp = va[:, 0] - vl
    tab_h, tab_V = np.array(case["pool_table"]["h"]), np.array(case["pool_table"]["V_mm3"])
    r.update(t=t, V_pool=Vp, V_lumen=vl, h_vol=np.interp(Vp * 1e9, tab_V, tab_h))
    w = case["stop_window_s"]
    r["q_pool"] = -moving_slope(t, Vp, w) * 6e7                       # mL/min, stop-test average
    r["q_pool_fast"] = -moving_slope(t, Vp, 0.05) * 6e7               # mL/min, short average
    to, qo = read_dat(run, "outletFlow")
    tq, ql = read_dat(run, "outletLiquid")
    if len(to) and len(tq):
        r["q_out_liq"] = np.interp(t, tq, ql[:, 0]) * 6e7
        r["q_out_gas"] = np.interp(t, to, qo[:, 0]) * 6e7 - r["q_out_liq"]
    ts, sec = read_dat(run, "tubeSection")
    if len(ts):
        r["p_tube"] = np.interp(t, ts, sec[:, 0])
    # probe lines -> local levels
    tp, a, xyz = read_probes(run, "levelProbes", "alpha.liquid")
    lines = case["level_lines"]
    if len(tp):
        n_exp = sum(ln["n"] for ln in lines)
        if a.shape[1] != n_exp:
            if xyz is None or len(xyz) != a.shape[1]:
                raise SystemExit("cannot match the level probes to the probe lines")
            g = np.array(case["gravity"])
            u = -g / np.linalg.norm(g)
            full = np.full((len(tp), n_exp), np.nan)
            k = 0
            for ln in lines:
                for i in range(ln["n"]):
                    p = (np.array(ln["start"]) + i * ln["ds"] * u) * 1e-3
                    j = np.argmin(np.linalg.norm(xyz - p, axis=1))
                    if np.linalg.norm(xyz[j] - p) < 1e-6:
                        full[:, k] = a[:, j]
                    k += 1
            a = full
        lev, k = [], 0
        for ln in lines:
            col = np.nan_to_num(a[:, k:k + ln["n"]])
            k += ln["n"]
            depth = col.sum(1) * ln["ds"]                               # liquid column above the floor
            lev.append(np.where(depth > 0.1, ln["h_floor"] + depth, np.nan))
        lev = np.array(lev).T                                           # (n_t, n_lines)
        r["t_probe"], r["h_lines"] = tp, lev
        with np.errstate(all="ignore"):
            r["h_probe"] = np.interp(t, tp, np.nanmedian(np.where(np.isfinite(lev), lev, np.nan), 1))
    # gas pressure in the drain: lumen probes that sit in gas
    tlp, pl, _ = read_probes(run, "lumenProbes", "p")
    _, al, _ = read_probes(run, "lumenProbes", "alpha.liquid")
    if len(tlp) and al.shape == pl.shape:
        n_top = len(case["lumen_probes"]) - 1                          # the last one is at the bore entrance
        pg = np.where(al[:, :n_top] < 0.5, pl[:, :n_top], np.nan)
        with np.errstate(all="ignore"):
            r["p_gas"] = np.interp(t, tlp, np.nanmedian(pg, 1))
    return r


def stop_levels(r):
    c = r["case"]
    t, h, q = r["t"], r["h_vol"], r["q_pool"]
    w, q_stop = c["stop_window_s"], c["q_stop_ml_min"]
    out = dict(t_end=float(t[-1]), h_end=float(np.mean(h[t >= t[-1] - 0.05])))
    if "h_probe" in r:
        out["h_end_probe"] = float(np.nanmean(r["h_probe"][t >= t[-1] - 0.05]))
    ok = (t >= w + 0.2) & np.isfinite(q) & (q < q_stop)
    if ok.any():
        i = int(np.argmax(ok))
        out.update(t_stop=float(t[i]), h_stop=float(h[i]))
        if "h_probe" in r:
            out["h_stop_probe"] = float(r["h_probe"][i])
    # straight line through Q(h) over the late, slow part of the drainage -> Q = 0
    sel = np.isfinite(q) & (q > q_stop) & (q < 30 * q_stop) & (t >= w + 0.2)
    if sel.sum() > 10 and np.ptp(h[sel]) > 0.2:
        k, d = np.polyfit(h[sel], q[sel], 1)
        if k > 0:
            out["h_stop_extrap"] = float(-d / k)
            out["extrap_from"] = [float(h[sel].min()), float(h[sel].max())]
    if "p_tube" in r:
        out["p_tube_end"] = float(np.mean(r["p_tube"][t >= t[-1] - 0.1]))
    if "p_gas" in r:
        out["p_gas_end"] = float(np.nanmean(r["p_gas"][t >= t[-1] - 0.1]))
    return out


def model_curve(case):
    from drainstop.model import DrainModel
    from run_study import build
    m = DrainModel(build(case["spec"]), case["tilt_deg"])
    d = m.descend(dh=0.05)
    q = d["qL"] * 6e7
    below = np.where(q < case["q_stop_ml_min"])[0]
    return dict(h=d["h"], q=q, pg=d["pg"], h_stop=d["h_stop"],
                h_q_stop=float(d["h"][below[0]]) if below.size else 0.0)


def summary(run, r, s, mc):
    c = r["case"]

    def f(x, nd=2):
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"
    lines = [
        f"drainstop VOF check: {c['name']} - {c['label']}, outlet end {c['tilt_deg']:g} deg up",
        f"run {run}",
        f"simulated {s['t_end']:.3f} s; started at h = {r['h_vol'][0]:.2f} mm (set {c['start']['full']:.2f} full / "
        f"{c['start']['short']:.2f} short); bore lip at {c['model']['lip']:.2f} mm",
        "",
        f"{'':44s}{'model':>10s}{'VOF':>10s}",
        f"{'stop level, Q < %g mL/min (mm)' % c['q_stop_ml_min']:44s}{f(mc['h_q_stop']):>10s}{f(s.get('h_stop')):>10s}"
        f"   (probe lines: {f(s.get('h_stop_probe'))})",
        f"{'stop level, Q -> 0 (mm)':44s}{f(mc['h_stop']):>10s}{f(s.get('h_stop_extrap')):>10s}"
        f"   (line through Q(h) for h = {f((s.get('extrap_from') or [None])[0])}..{f((s.get('extrap_from') or [None, None])[1])})",
        f"{'pool level at the end of the run (mm)':44s}{'':>10s}{f(s['h_end']):>10s}   (probe lines: {f(s.get('h_end_probe'))})",
        f"{'pool drain rate at the end (mL/min)':44s}{'':>10s}{f(float(r['q_pool'][-1]), 1):>10s}",
        f"{'gas pressure in the drain at the end (Pa)':44s}{f(float(np.interp(s['h_end'], mc['h'][::-1], mc['pg'][::-1])), 0):>10s}"
        f"{f(s.get('p_gas_end'), 0):>10s}",
        f"{'static pressure in the tube, 2 bore d (Pa)':44s}{f(c['p_outlet'], 0):>10s}{f(s.get('p_tube_end'), 0):>10s}",
        "",
        "Stop level = pool level h above the lowest point of the skin, from the pool's liquid volume.",
        "STOPPED / FINISHED in the run directory say why the run ended.",
    ]
    return "\n".join(lines)


def plots(run, r, s, mc, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - no plots")
        return
    c = r["case"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(r["t"], r["h_vol"], label="pool volume")
    if "h_lines" in r:
        ax[0].plot(r["t_probe"], r["h_lines"], lw=0.5, color="0.6")
        ax[0].plot(r["t"], r["h_probe"], label="probe lines (median)")
    ax[0].axhline(mc["h_stop"], ls="--", color="k", label=f"model stop {mc['h_stop']:.2f} mm")
    ax[0].axhline(c["model"]["lip"], ls=":", color="r", label="bore lip")
    ax[0].set(xlabel="t (s)", ylabel="pool level h (mm)", title=c["name"])
    ax[0].legend(fontsize=8)
    ax[1].plot(r["h_vol"], r["q_pool"], label="VOF, pool drain rate")
    if "q_out_liq" in r:
        ax[1].plot(r["h_vol"], r["q_out_liq"], lw=0.5, alpha=0.6, label="VOF, liquid at the outlet")
    ax[1].plot(mc["h"], mc["q"], "k--", label="model Q_L(h)")
    ax[1].set(xlabel="pool level h (mm)", ylabel="flow (mL/min)", ylim=(0, None))
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=8)
    if "p_gas" in r:
        ax[2].plot(r["h_vol"], r["p_gas"], label="VOF, gas in the drain")
    if "p_tube" in r:
        ax[2].plot(r["h_vol"], r["p_tube"], lw=0.6, label="VOF, tube section")
    ax[2].plot(mc["h"], mc["pg"], "k--", label="model p_g(h)")
    ax[2].set(xlabel="pool level h (mm)", ylabel="pressure (Pa gauge)")
    ax[2].invert_xaxis()
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=120)


def timing(run):
    """Seconds per step and time step from the interFoam logs; projected wall time to the stop."""
    case = json.load(open(os.path.join(run, "case.json")))
    clock, sim, dts = [], [], []
    for f in sorted(glob.glob(os.path.join(run, "log.interFoam.*")), key=os.path.getmtime):
        cl, tm, dt = [], [], []
        for line in open(f, errors="replace"):
            if line.startswith("deltaT ="):
                dt.append(float(line.split("=")[1]))
            elif line.startswith("Time ="):
                tm.append(_num(line.split("=")[1].strip()))
            elif line.startswith("ExecutionTime"):
                m = re.search(r"ExecutionTime = ([\d.eE+-]+) s\s+ClockTime = ([\d.eE+-]+)", line)
                cl.append((float(m.group(1)), float(m.group(2))))
        n = min(len(cl), len(tm), len(dt))
        clock.append(np.array(cl[:n]).reshape(-1, 2))
        sim += tm[:n]
        dts += dt[:n]
    if not dts or len(clock[-1]) < 3:
        raise SystemExit("no interFoam log with enough time steps in " + run)
    cells = None
    if os.path.exists(os.path.join(run, "log.checkMesh")):
        m = re.search(r"^\s*cells:\s+(\d+)", open(os.path.join(run, "log.checkMesh")).read(), re.M)
        cells = int(m.group(1)) if m else None
    np_ = len(glob.glob(os.path.join(run, "processor*"))) or 1
    last = clock[-1]
    k = max(3, len(last) // 5)
    span = last[-1] - last[-k]                      # (ExecutionTime, ClockTime) over the last k steps
    # ClockTime is whole seconds: use it when the span is long enough, else the CPU time of the master
    s_step = float((span[1] if span[1] >= 30 else span[0]) / (k - 1))
    dt_now = float(np.median(dts[-k:]))
    print(f"run {run}")
    print(f"  cells {cells}, {np_} processes, {len(dts)} steps, simulated {sim[-1]:.5f} s")
    print(f"  {s_step:.3f} s per step (last {k} steps), time step {dt_now * 1e6:.2f} us now")
    if cells:
        print(f"  cost {s_step * np_ / cells * 1e6:.1f} us per cell, step and core "
              f"(the estimate in vof/runtime_estimate.md assumes 10 at 80 % efficiency)")
    print(f"  {dt_now / s_step * 3600:.3f} s of simulated time per wall-clock hour at this time step")
    try:
        r = load(run)
        h_now = float(r["h_vol"][-1])
        mc = model_curve(case)
        tab_h, tab_V = np.array(case["pool_table"]["h"]), np.array(case["pool_table"]["V_mm3"])
        A = np.gradient(tab_V, tab_h)
        sel = (mc["h"] <= h_now) & (mc["q"] > case["q_stop_ml_min"])
        hh, qq = mc["h"][sel], mc["q"][sel] * 1e3 / 60
        if len(hh) > 1:
            rate = np.interp(hh, tab_h, A) / qq
            t_rest = float(np.sum(0.5 * (rate[1:] + rate[:-1]) * -np.diff(hh))) + case["stop_window_s"]
            print(f"  pool at {h_now:.2f} mm; the model needs {t_rest:.2f} s more to reach "
                  f"{case['q_stop_ml_min']:g} mL/min at {hh[-1]:.2f} mm -> about "
                  f"{t_rest / dt_now * s_step / 3600:.0f} h of wall time at this step size")
        else:
            print(f"  pool at {h_now:.2f} mm, at or below the model's stop level")
    except SystemExit as e:
        print(" ", e)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.timing:
        timing(a.run)
        return
    r = load(a.run)
    s = stop_levels(r)
    mc = model_curve(r["case"])
    out = os.path.join(a.run, "analysis")
    os.makedirs(out, exist_ok=True)
    txt = summary(a.run, r, s, mc)
    print(txt)
    open(os.path.join(out, "summary.txt"), "w").write(txt + "\n")
    json.dump(dict(vof=s, model=dict(h_stop=mc["h_stop"], h_q_stop=mc["h_q_stop"])),
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    np.savetxt(os.path.join(out, "series.csv"),
               np.column_stack([r["t"], r["h_vol"], r.get("h_probe", np.full(len(r["t"]), np.nan)), r["q_pool"],
                                r.get("q_out_liq", np.full(len(r["t"]), np.nan)),
                                r.get("q_out_gas", np.full(len(r["t"]), np.nan)),
                                r.get("p_gas", np.full(len(r["t"]), np.nan)),
                                r.get("p_tube", np.full(len(r["t"]), np.nan))]),
               delimiter=",", fmt="%.6g",
               header="t_s,h_vol_mm,h_probe_mm,q_pool_ml_min,q_out_liquid_ml_min,q_out_gas_ml_min,p_gas_Pa,p_tube_Pa")
    if a.plot:
        plots(a.run, r, s, mc, out)


if __name__ == "__main__":
    main()

"""Write LaTeX tables and number macros for the paper from results/*.json.

    py -3.12 scripts/make_tables.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from drainstop.model import Drain, Fluid, Gas, LAYOUTS, PATTERN_KEYS, MMHG, cd_hole  # noqa: E402
import run_study  # noqa: E402

RES = os.path.join(ROOT, "results")
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)
TILTS = [-10, -5, 0, 5, 10, 15, 20]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return [r for r in json.load(f) if "error" not in r] if name != "recommended.json" else json.load(f)


def write(name, text):
    with open(os.path.join(GEN, name), "w", encoding="utf-8") as f:
        f.write(text)
    print("  ", name)


def tilt_head():
    return " & ".join(f"${t:+d}^\\circ$" if t else "$0^\\circ$" for t in TILTS)


def fmt(v):
    return f"{v:.1f}"


def numbers():
    dr, fl, gas = Drain(), Fluid(), Gas()
    ps = 7.0 * MMHG
    Hv = ps / (fl.rho * 9.81) * 1e3
    Ap = dr.port_area * 1e6
    qmax = 0.85 * dr.port_area * math.sqrt(2 * ps / fl.rho) * 6e7
    dstar = 4 * fl.gamma / ps * 1e3
    sw = load("sweep.json")
    err = np.array([r["h_sim"] - r["h_ana"] for r in sw])
    err_s = np.array([r["h_sim"] - r["h_ana_simple"] for r in sw])
    mc = load("montecarlo.json")
    rec = load("recommended.json")
    tl = load("tiltlimit.json")
    rr = {r["tilt"]: r for r in tl if r["design"] == "recommended"}
    lim = 0.0
    for t in sorted(t for t in rr if t >= 0):
        if rr[t]["h_sim"] - rr[t]["h_geo"] <= 0.1:
            lim = t
        else:
            break
    sz = load("sizes_rec.json")
    r0 = next(r for r in sz if abs(r["d"] - 1.0) < 1e-9 and r["tilt"] == 0)
    rec_h = [next(r for r in sz if abs(r["d"] - 1.0) < 1e-9 and r["tilt"] == t)["h_sim"] for t in TILTS]
    usz = {r["tilt"]: r for r in sw if r["layout"] == "usz"}
    m = {
        "Hv": f"{Hv:.1f}", "psPa": f"{ps:.0f}", "Ap": f"{Ap:.2f}", "Qmax": f"{qmax:.0f}",
        "dstar": f"{dstar:.2f}", "PcHalf": f"{4 * fl.gamma / 0.5e-3:.0f}", "PcOne": f"{4 * fl.gamma / 1e-3:.0f}",
        "PcFour": f"{4 * fl.gamma / 4e-3:.0f}", "zlip": f"{dr.b - dr.port_d / 2:.2f}",
        "nSweep": f"{len(sw)}", "errMean": f"{np.mean(np.abs(err)):.2f}", "errMax": f"{np.max(np.abs(err)):.2f}",
        "errMeanS": f"{np.mean(np.abs(err_s)):.2f}", "errMaxS": f"{np.max(np.abs(err_s)):.1f}",
        "recQ": f"{r0['q0'] * 6e7:.0f}", "recQfrac": f"{100 * r0['q0'] / r0['q_port']:.0f}",
        "recN": f"{r0['n']}", "recArea": f"{r0['area']:.1f}", "recLam": f"{r0['Lambda_tot']:.2f}",
        "recTiltLimit": f"{lim:.1f}", "recHmin": f"{min(rec_h):.1f}", "recHmax": f"{max(rec_h):.1f}",
        "uszZero": fmt(usz[0]["h_sim"]), "uszTen": fmt(usz[10]["h_sim"]), "uszTwenty": fmt(usz[20]["h_sim"]),
        "uszArea": f"{usz[0]['area']:.0f}", "uszLam": f"{usz[0]['Lambda_tot']:.0f}",
        "cdh": f"{float(cd_hole(dr.wall / 1.0)):.2f}",
    }
    for key, name in (("recommended", "Rec"), ("usz", "Usz"), ("side2", "Side"), ("cross4", "Cross")):
        v = np.array([r["h_sim"] for r in mc if r["design"] == key])
        g = np.array([r["h_sim"] - r["h_geo"] for r in mc if r["design"] == key])
        m[f"mc{name}Med"] = f"{np.median(v):.1f}"
        m[f"mc{name}Pnf"] = f"{np.percentile(v, 95):.1f}"
        m[f"mc{name}Full"] = f"{100 * np.mean(g <= 0.1):.0f}"
    m["nMC"] = f"{len([r for r in mc if r['design'] == 'recommended'])}"
    mcap = {r["tilt"]: r for r in load("mincap.json")}
    for t, name in ((5, "Five"), (10, "Ten"), (15, "Fifteen"), (20, "Twenty")):
        m[f"cap{name}"] = f"{mcap[t]['q_min_lpm']:.0f}"
    m["airLo"] = f"{min(r['q_air_peak_lpm'] for r in mcap.values()):.0f}"
    m["airHi"] = f"{max(r['q_air_peak_lpm'] for r in mcap.values()):.0f}"
    txt = "".join(f"\\newcommand{{\\num{k}}}{{{v}}}\n" for k, v in m.items())
    write("numbers.tex", txt)


def tab_layouts():
    sw = load("sweep.json")
    sz = load("sizes_rec.json")
    lines = [r"\begin{tabular}{@{}l r r " + "r" * len(TILTS) + "@{}}", r"\toprule",
             r" & & & \multicolumn{" + str(len(TILTS)) + r"}{c}{stop level $h_s$ (mm) at tilt $\alpha$} \\",
             r"\cmidrule(l){4-" + str(3 + len(TILTS)) + "}",
             r"layout & $A$ (mm$^2$) & $Q_0$ (mL/min) & " + tilt_head() + r" \\", r"\midrule"]

    def row(label, rr):
        r0 = rr[0]
        vals = " & ".join(fmt(rr[t]["h_sim"]) for t in TILTS)
        return f"{label} & {r0['area']:.0f} & {r0['q0'] * 6e7:.0f} & {vals} \\\\"
    lines.append(row("USZ as built (318 $\\times$ \\O1.0)", {r["tilt"]: r for r in sw if r["layout"] == "usz"}))
    for i, k in enumerate(PATTERN_KEYS):
        rr = {r["tilt"]: r for r in sw if r["layout"] == k and r["family"] == "fixed_area" and abs(r["d"] - 1) < 1e-9}
        lines.append(row(f"{chr(65 + i)}\\ \\ {LAYOUTS[k][0].replace(' deg', '$^\\circ$')}", rr))
    rr = {r["tilt"]: r for r in sz if abs(r["d"] - 1) < 1e-9}
    lines.append(r"\midrule")
    lines.append(row(r"\textbf{Recommended}, $\pm30^\circ$ staggered", rr))
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_layouts.tex", "\n".join(lines) + "\n")


def tab_sizes():
    sz = load("sizes_rec.json")
    lines = [r"\begin{tabular}{@{}r r r r " + "r" * len(TILTS) + "@{}}", r"\toprule",
             r"\O\ (mm) & holes & $A$ (mm$^2$) & $Q_0$ (mL/min) & " + tilt_head() + r" \\", r"\midrule"]
    for d in sorted(set(r["d"] for r in sz)):
        rr = {r["tilt"]: r for r in sz if r["d"] == d}
        r0 = rr[0]
        vals = " & ".join(fmt(rr[t]["h_sim"]) for t in TILTS)
        bold = r"\textbf{" if abs(d - 1.0) < 1e-9 else "{"
        lines.append(f"{bold}{d:g}}} & {r0['n']} & {r0['area']:.0f} & {r0['q0'] * 6e7:.0f} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sizes.tex", "\n".join(lines) + "\n")


def tab_validation():
    tr = load("transient.json")
    names = {"recommended": "Recommended", "usz": "USZ as built", "side2": "Side rows", "cross4": "Four rows"}
    lines = [r"\begin{tabular}{@{}l r r r r r@{}}", r"\toprule",
             r"design & $\alpha$ & closed form & network & time-dependent & lowest hole edge \\",
             r" & & Eq.~\eqref{eq:stop-refined} & (quasi-static) & simulation & $h_{geo}$ \\", r"\midrule"]
    for r in sorted(tr, key=lambda r: (list(names).index(r["design"]), r["tilt"])):
        lines.append(f"{names[r['design']]} & ${r['tilt']}^\\circ$ & {r['h_ana']:.2f} & {r['h_sim']:.2f} & "
                     f"{r['h_dyn']:.2f} & {r['h_geo']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_validation.tex", "\n".join(lines) + "\n")


def tab_pump():
    pp = load("pump.json")
    fl, gas = Fluid(), Gas()
    caps = [2.0, 5.0, 10.0, 20.0, math.inf]
    lines = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
             r"source capacity & \multicolumn{2}{c}{largest open area that still drains fully (mm$^2$)} & "
             r"\multicolumn{2}{c}{Eq.~\eqref{eq:cap-area} (mm$^2$)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r" & $\alpha=+10^\circ$ & $\alpha=+15^\circ$ & $+10^\circ$ & $+15^\circ$ \\", r"\midrule"]
    m10 = run_study.make_model(dict(layout="custom", angle=30.0, d=1.0, n_x=18, family="custom", tilt=10))
    m15 = run_study.make_model(dict(layout="custom", angle=30.0, d=1.0, n_x=18, family="custom", tilt=15))
    for c in caps:
        cells = []
        for t in (10, 15):
            rr = sorted([r for r in pp if r["q_max_lpm"] == c and r["tilt"] == t], key=lambda r: r["area"])
            ok = [r["area"] for r in rr if r["h_sim"] - r["h_geo"] <= 0.1]
            cells.append(f"{max(ok):.0f}" if ok else "--")
        est = []
        for m in (m10, m15):
            if math.isinf(c):
                est.append("--")
                continue
            dz = (m.lip_height() - 1.5) * 1e-3
            v = math.sqrt(2 * fl.rho * 9.81 * dz / gas.rho)
            est.append(f"{(c * 1e-3 / 60) / (0.61 * v) * 1e6:.0f}")
        lab = "ideal regulator" if math.isinf(c) else f"{c:g} L/min"
        lines.append(f"{lab} & {cells[0]} & {cells[1]} & {est[0]} & {est[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_pump.tex", "\n".join(lines) + "\n")


def tab_sensitivity():
    se = load("sensitivity.json")
    cases = list(dict.fromkeys(r["case"] for r in se))
    des = ["recommended", "usz", "side2", "cross4"]
    head = " & ".join([r"\multicolumn{2}{c}{" + n + "}" for n in ("Recommended", "USZ as built", "Side rows", "Four rows")])
    lines = [r"\begin{tabular}{@{}l " + "r r " * 4 + "@{}}", r"\toprule", r"case & " + head + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(4)),
             r" & " + " & ".join([r"$0^\circ$ & $+10^\circ$"] * 4) + r" \\", r"\midrule"]
    for c in cases:
        cells = []
        for d in des:
            for t in (0, 10):
                r = next(x for x in se if x["design"] == d and x["case"] == c and x["tilt"] == t)
                cells.append(fmt(r["h_sim"]))
        lines.append(f"{c.replace('%', r'\%')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sensitivity.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    numbers()
    tab_layouts()
    tab_sizes()
    tab_validation()
    tab_pump()
    tab_sensitivity()

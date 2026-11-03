"""Run the drain-stop study and write results/*.json.

    py -3.12 scripts/run_study.py [sweep] [curves] [optimize] [angles] [sensitivity] [montecarlo] [transient]
                                  [tiltlimit] [pump] [mincap] [sizes_rec]

No argument runs everything. Uses all cores but two.
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from drainstop.model import (Drain, DrainModel, FLUIDS, Options, Suction, MMHG, PATTERN_KEYS,  # noqa: E402
                             Pattern, rows_pattern, fixed_area_pattern, usz_as_built, Ellipse)
from drainstop import transient  # noqa: E402

OUT = os.path.join(ROOT, "results")
SIZES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TILTS = [-10, -5, 0, 5, 10, 15, 20]
N_X = 19                                   # stations per row in the fixed-count family
A_TARGET = 2.0 * Drain().port_area * 1e6   # mm2, open area of the fixed-area family (2 x bore)
WORKERS = max(1, (os.cpu_count() or 2) - 2)


def build(spec):
    if spec["layout"] == "usz":
        return usz_as_built()
    if spec["layout"] == "custom":
        return custom_two_rows(spec["angle"], spec["d"], spec["n_x"], spec.get("stagger", 0.5))
    if spec["family"] == "fixed_area":
        return fixed_area_pattern(spec["layout"], spec["d"], spec.get("area", A_TARGET))
    if spec["family"] == "fixed_n":
        return rows_pattern(spec["layout"], spec["d"], spec["n_x"], family="fixed_n")
    return rows_pattern(spec["layout"], spec["d"], N_X, family="fixed_count")


def custom_two_rows(angle_deg, d, n_x, stagger=0.5):
    """Two rows at +-angle (staggered by default) with n_x holes per row."""
    drain = Drain()
    pitch = (drain.x_hi - drain.x_lo) / n_x
    xs, ths = [], []
    for sgn, off in ((1, 0.0), (-1, stagger)):
        x = drain.x_lo + ((np.arange(n_x) + 0.5 + off) % n_x) * pitch
        xs.append(np.sort(x))
        ths.append(np.full(n_x, sgn * math.radians(angle_deg)))
    x = np.concatenate(xs)
    return Pattern("custom", f"Two rows at +-{angle_deg:g} deg", x, np.concatenate(ths),
                   np.full(x.size, float(d)), np.ones(x.size), (angle_deg, -angle_deg), "custom")


def make_model(spec):
    opts = Options(**spec.get("opts", {}))
    fluid = FLUIDS[spec.get("fluid", "gly50")]
    suction = Suction(p_set=spec.get("p_mmhg", -7.0) * MMHG, q_max=spec.get("q_max", math.inf))
    drain = Drain(**spec.get("drain", {}))
    return DrainModel(build(spec), spec["tilt"], drain=drain, fluid=fluid, suction=suction, opts=opts)


def run_case(spec, with_curve=False, sim=True):
    m = make_model(spec)
    p = m.pattern
    a = m.analytic(dh=0.02, with_capillary=spec.get("opts", {}).get("theta_e", 90) < 90)
    res = dict(spec)
    res.update(n=p.n, area=p.open_area_mm2, lip=m.lip_height(), h_geo=a["h_geo"],
               h_ana=a["h_stop"], h_ana_simple=a["h_stop_simple"], q_port=m.q_port_max(),
               Lambda_tot=float(np.sum(m.cd * m.A) / (m.opts.cd_port * m.Ap)))
    if sim:
        r = m.descend(dh=0.1)
        hs, t = r["h"], r["t"]
        ok = np.isfinite(t)
        t_stop = float(np.interp(r["h_stop"] + 0.25, hs[ok][::-1], t[ok][::-1])) if ok.sum() > 1 else 0.0
        res.update(h_sim=r["h_stop"], q0=r["q0"], t_stop=t_stop)
        if with_curve:
            res["curve"] = {k: r[k].tolist() for k in ("h", "qL", "qG", "pg", "zeta", "Lambda", "t", "mode")}
            res["curve_ana"] = {k: a[k].tolist() for k in ("h", "Lambda", "Lambda_w", "lift", "lift_simple")}
    else:
        res["q0"] = m.state(m.h_start() + m.Z_min)["qL"]
    return res


def _run(args):
    spec, with_curve, sim = args
    try:
        return run_case(spec, with_curve, sim)
    except Exception as e:  # keep the sweep going, report the case
        r = dict(spec)
        r["error"] = repr(e)
        return r


def pmap(specs, with_curve=False, sim=True):
    with Pool(WORKERS) as pool:
        return pool.map(_run, [(s, with_curve, sim) for s in specs], chunksize=1)


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print("  wrote", name)


# ----------------------------------------------------------------------------- parts
def sweep():
    specs = [dict(layout=k, d=d, family=fam, tilt=t)
             for k in PATTERN_KEYS for d in SIZES for fam in ("fixed_count", "fixed_area") for t in TILTS]
    specs += [dict(layout="usz", d=1.0, family="reference", tilt=t) for t in TILTS]
    save("sweep.json", pmap(specs))


def curves():
    specs = [dict(layout=k, d=1.0, family="fixed_area", tilt=t) for k in PATTERN_KEYS for t in (0, 10)]
    specs += [dict(layout="usz", d=1.0, family="reference", tilt=t) for t in (0, 10)]
    specs += [dict(best_spec(), tilt=t, design="recommended") for t in (0, 10)]
    save("curves.json", pmap(specs, with_curve=True))


def optimize():
    """Two staggered rows at +-30 deg: map of hole size x holes per row, over the tilt range."""
    sizes = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
    specs = []
    for d in sizes:
        n_max = min(60, int((Drain().x_hi - Drain().x_lo) // (d + 0.6)))
        for n_x in sorted(set(np.unique(np.round(np.geomspace(1, n_max, 16)).astype(int)))):
            for t in TILTS:
                specs.append(dict(layout="custom", angle=30.0, d=d, n_x=int(n_x), family="custom", tilt=t))
    save("optimize.json", pmap(specs, sim=False))


def angles():
    specs = [dict(layout="custom", angle=ang, d=1.0, n_x=14, family="custom", tilt=t,
                  opts=dict(floor_gap=fg))
             for ang in (15, 22.5, 30, 37.5, 45, 60, 75, 90, 120, 150) for t in TILTS for fg in (0.0, 0.2)]
    save("angles.json", pmap(specs))


def best_spec():
    with open(os.path.join(OUT, "recommended.json")) as f:
        return json.load(f)


def sensitivity():
    rec = best_spec()
    designs = {
        "usz": dict(layout="usz", d=1.0, family="reference"),
        "side2": dict(layout="side2", d=1.0, family="fixed_area"),
        "cross4": dict(layout="cross4", d=1.0, family="fixed_area"),
        "recommended": dict(rec),
    }
    cases = {
        "baseline": {},
        "capillary 60 deg": dict(opts=dict(theta_e=60.0)),
        "capillary 0 deg": dict(opts=dict(theta_e=0.0)),
        "drawdown 0.5 d": dict(opts=dict(kappa=0.5)),
        "pump 2 L/min": dict(q_max=2.0e-3 / 60),
        "pump 5 L/min": dict(q_max=5.0e-3 / 60),
        "water": dict(fluid="water"),
        "80% glycerol": dict(fluid="gly80"),
        "Cd holes x0.8": dict(opts=dict(cd_scale=0.8)),
        "Cd port 0.62": dict(opts=dict(cd_port=0.62)),
        "-5 mmHg": dict(p_mmhg=-5.0),
        "-10 mmHg": dict(p_mmhg=-10.0),
        "on the floor": dict(opts=dict(floor_gap=0.2)),
    }
    specs = []
    for dn, ds in designs.items():
        for cn, cs in cases.items():
            for t in (0, 10):
                s = dict(ds)
                s.update(tilt=t, design=dn, case=cn)
                for k, v in cs.items():
                    if k == "opts":
                        s["opts"] = dict(s.get("opts", {}), **v)
                    else:
                        s[k] = v
                specs.append(s)
    save("sensitivity.json", pmap(specs))


def montecarlo(n=200, seed=7):
    rng = np.random.default_rng(seed)
    rec = best_spec()
    designs = {
        "usz": dict(layout="usz", d=1.0, family="reference"),
        "side2": dict(layout="side2", d=1.0, family="fixed_area"),
        "cross4": dict(layout="cross4", d=1.0, family="fixed_area"),
        "recommended": dict(rec),
    }
    specs = []
    for i in range(n):
        draw = dict(tilt=float(rng.uniform(-10, 15)),
                    p_mmhg=float(rng.uniform(-8.0, -6.0)),
                    fluid=str(rng.choice(["water", "gly40", "gly50", "gly80"])),
                    opts=dict(cd_scale=float(rng.uniform(0.85, 1.15)),
                              cd_port=float(rng.uniform(0.62, 0.9)),
                              theta_e=float(rng.uniform(60, 90)),
                              kappa=float(rng.uniform(0, 0.5))))
        for dn, ds in designs.items():
            s = dict(ds)
            s.update(draw, design=dn, sample=i)
            s["opts"] = dict(ds.get("opts", {}), **draw["opts"])
            specs.append(s)
    save("montecarlo.json", pmap(specs))


def transient_runs():
    rec = best_spec()
    designs = {
        "usz": dict(layout="usz", d=1.0, family="reference"),
        "side2": dict(layout="side2", d=1.0, family="fixed_area"),
        "cross4": dict(layout="cross4", d=1.0, family="fixed_area"),
        "recommended": dict(rec),
    }
    specs = [dict(ds, tilt=t, design=dn) for dn, ds in designs.items() for t in (0, 10)]
    with Pool(WORKERS) as pool:
        out = pool.map(_transient, specs, chunksize=1)
    save("transient.json", out)


def _transient(spec):
    m = make_model(spec)
    r = transient.simulate(m, t_end=1800.0)
    a = m.analytic(dh=0.02)
    q = m.descend(dh=0.1)
    res = dict(spec)
    res.update(h_dyn=r["h_final"], h_sim=q["h_stop"], h_ana=a["h_stop"], h_geo=a["h_geo"],
               lip=m.lip_height(), trace={k: v.tolist() for k, v in r["trace"].items()})
    return res


def tiltlimit():
    """Stop level versus tilt up to 40 deg for the main designs (outlet up)."""
    rec = best_spec()
    designs = {"usz": dict(layout="usz", d=1.0, family="reference"), "recommended": dict(rec)}
    for k in PATTERN_KEYS:
        designs[k] = dict(layout=k, d=1.0, family="fixed_area")
    specs = [dict(ds, tilt=float(t), design=dn) for dn, ds in designs.items()
             for t in np.arange(-15, 40.1, 2.5)]
    save("tiltlimit.json", pmap(specs))


def pump():
    """Source with limited flow capacity: stop level versus open area (two rows at +-30 deg, 1.0 mm)."""
    caps = [2.0, 5.0, 10.0, 20.0, math.inf]                       # L/min
    specs = [dict(layout="custom", angle=30.0, d=1.0, n_x=int(n), family="custom", tilt=t,
                  q_max=(c * 1e-3 / 60 if math.isfinite(c) else math.inf), q_max_lpm=c)
             for c in caps for t in (0, 5, 10, 15) for n in (2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 27, 33, 40)]
    save("pump.json", pmap(specs))


def _min_capacity(tilt):
    """Smallest source capacity (L/min) that still drains the recommended layout to its lowest hole."""
    rec = best_spec()

    def drains(c):
        m = make_model(dict(rec, tilt=tilt, q_max=(c * 1e-3 / 60 if math.isfinite(c) else math.inf)))
        r = m.descend(dh=0.1)
        return r["h_stop"] - m.analytic(dh=0.02)["h_geo"] <= 0.1, r

    ok, r = drains(math.inf)
    res = dict(tilt=tilt, q_air_peak_lpm=float(np.max(r["qG"])) * 6e4, q_min_lpm=None)
    if ok:
        lo, hi = 1.0, 40.0
        for _ in range(9):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if drains(mid)[0] else (mid, hi)
        res["q_min_lpm"] = hi
    return res


def mincap():
    """Flow a regulator must pass at -7 mmHg for the recommended layout to drain fully, per tilt."""
    with Pool(WORKERS) as pool:
        save("mincap.json", pool.map(_min_capacity, [0.0, 5.0, 10.0, 15.0, 20.0], chunksize=1))


def sizes_rec():
    """Recommended layout (two staggered rows at +-30 deg) at about 28 mm2 for every hole size."""
    specs = []
    for d in SIZES:
        n_x = max(1, int(round(A_TARGET / (2 * math.pi * d ** 2 / 4))))
        specs += [dict(layout="custom", angle=30.0, d=d, n_x=n_x, family="custom", tilt=t) for t in TILTS]
    save("sizes_rec.json", pmap(specs))


PARTS = {"sweep": sweep, "curves": curves, "optimize": optimize, "angles": angles,
         "sensitivity": sensitivity, "montecarlo": montecarlo, "transient": transient_runs,
         "tiltlimit": tiltlimit, "pump": pump, "mincap": mincap, "sizes_rec": sizes_rec}

if __name__ == "__main__":
    todo = sys.argv[1:] or list(PARTS)
    for name in todo:
        t0 = time.time()
        print(f"[{name}]")
        PARTS[name]()
        print(f"  {time.time() - t0:.0f} s")

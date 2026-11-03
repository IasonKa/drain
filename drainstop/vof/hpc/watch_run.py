"""Watch an interFoam run of the drainstop VOF check and stop it cleanly.

Started by run_vof.sbatch beside interFoam (standard library only). Every --every-s seconds it reads the
liquid volumes that interFoam writes (postProcessing/liquidVolume and lumenVolume; pool = all - lumen),
appends a status line to stdout (the job sends it to <run>/watch.log), and

  - writes STOPPED and sets stopAt writeNow once the pool has stopped draining: over the last
    stop_window_s of simulated time (case.json) the pool lost liquid slower than q_stop_ml_min,
    and at least stop_window_s + 0.2 s have been simulated;
  - writes WALLTIME and sets stopAt writeNow when the job is --margin-s from its time limit.

interFoam then writes the current time and exits (runTimeModifiable).

    python vof/hpc/watch_run.py <run dir> --walltime-s 86400 [--margin-s 1800] [--every-s 60]
"""
import argparse
import bisect
import glob
import json
import os
import re
import sys
import time


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def read_series(run, name):
    """(times, values) of the first data column of a function object, all restarts joined."""
    data = {}
    base = os.path.join(run, "postProcessing", name)
    dirs = [d for d in glob.glob(os.path.join(base, "*")) if _num(os.path.basename(d)) is not None]
    for d in sorted(dirs, key=lambda p: float(os.path.basename(p))):
        for f in sorted(glob.glob(os.path.join(d, "*.dat"))):
            with open(f) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    p = line.split()
                    if len(p) >= 2 and _num(p[0]) is not None and _num(p[1]) is not None:
                        data[float(p[0])] = float(p[1])
    ts = sorted(data)
    return ts, [data[t] for t in ts]


def pool_volume(run):
    ta, va = read_series(run, "liquidVolume")
    tl, vl = read_series(run, "lumenVolume")
    lum = dict(zip(tl, vl))
    t, v = [], []
    for ti, vi in zip(ta, va):
        if ti in lum:
            t.append(ti)
            v.append(vi - lum[ti])
    return t, v


def slope(t, v):
    n = len(t)
    mt, mv = sum(t) / n, sum(v) / n
    den = sum((x - mt) ** 2 for x in t)
    return sum((x - mt) * (y - mv) for x, y in zip(t, v)) / den if den > 0 else 0.0


def level_of(case, v_m3):
    """Equivalent pool level (mm) from the pool volume table in case.json."""
    hs, vs = case["pool_table"]["h"], case["pool_table"]["V_mm3"]
    v = v_m3 * 1e9
    i = min(max(bisect.bisect_left(vs, v), 1), len(vs) - 1)
    f = (v - vs[i - 1]) / (vs[i] - vs[i - 1]) if vs[i] > vs[i - 1] else 0.0
    return hs[i - 1] + f * (hs[i] - hs[i - 1])


def stop_now(run, marker, info):
    with open(os.path.join(run, marker), "w") as f:
        json.dump(info, f, indent=1)
    cd = os.path.join(run, "system", "controlDict")
    with open(cd) as f:
        txt = f.read()
    txt = re.sub(r"^(\s*stopAt\s+)\w+;", r"\1writeNow;", txt, count=1, flags=re.M)
    with open(cd, "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--walltime-s", type=float, default=86400)
    ap.add_argument("--margin-s", type=float, default=1800)
    ap.add_argument("--every-s", type=float, default=60)
    a = ap.parse_args()
    t0 = time.time()
    case = json.load(open(os.path.join(a.run, "case.json")))
    window = case["stop_window_s"]
    q_stop = case["q_stop_ml_min"] * 1e-6 / 60          # m3/s
    print(f"# watch_run started {time.strftime('%Y-%m-%d %H:%M:%S')}: stop when the pool drains slower than "
          f"{case['q_stop_ml_min']:g} mL/min over {window:g} s; wall time {a.walltime_s:.0f} s, margin {a.margin_s:.0f} s")
    print("# wall_s  sim_t_s  h_pool_mm  pool_drain_ml_min")
    sys.stdout.flush()
    while True:
        time.sleep(a.every_s)
        t, v = pool_volume(a.run)
        status = None
        if len(t) > 5:
            i = bisect.bisect_left(t, t[-1] - window)
            q = -slope(t[i:], v[i:]) if len(t) - i > 5 else float("nan")
            h = level_of(case, v[-1])
            print(f"{time.time() - t0:8.0f} {t[-1]:9.4f} {h:9.3f} {q * 6e7:10.2f}")
            sys.stdout.flush()
            if t[-1] >= window + 0.2 and t[-1] - t[0] >= window and q < q_stop:
                status = ("STOPPED", dict(t=t[-1], h_pool_mm=h, pool_drain_ml_min=q * 6e7,
                                          reason=f"pool drains slower than {case['q_stop_ml_min']:g} mL/min "
                                                 f"over the last {window:g} s"))
        if status is None and time.time() - t0 > a.walltime_s - a.margin_s:
            status = ("WALLTIME", dict(t=t[-1] if t else None, reason="job time limit"))
        if status:
            stop_now(a.run, *status)
            print(f"# {status[0]}: {status[1]['reason']} - stopAt writeNow set")
            return


if __name__ == "__main__":
    main()

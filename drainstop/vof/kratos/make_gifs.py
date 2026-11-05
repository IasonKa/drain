"""Slow, captioned animations of a Kratos drain run (one frame per field output, every 5 ms).

    python vof/kratos/make_gifs.py <run dir> [--seconds 1.5]

Writes <run>/pictures/midplane_slow.gif (whole drain) and outlet_end_slow.gif (close-up with
streamlines). Each frame's title gives the time and the measured outflows and pressure at that time.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_kratos as rk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seconds", type=float, default=1.5, help="display time per frame")
    a = ap.parse_args()
    import imageio.v2 as imageio
    from PIL import Image
    run = os.path.abspath(a.run)
    st = json.load(open(os.path.join(run, "model_state.json")))
    mon = rk.read_monitor(run)
    frame = rk.Frame(rk.TILT)
    files = rk.vtk_files(run)
    label = "USZ as built" if st["case"].startswith("usz") else "Recommended"
    out = os.path.join(run, "pictures", "frames_slow")
    os.makedirs(out, exist_ok=True)
    gifs = {"midplane": [], "outlet_end": []}
    for f in files:
        t = rk.vtk_time(f)
        g = rk.load(f)
        i = int(np.argmin(np.abs(mon["t"] - t)))
        ql = mon["Q_liquid"][i] * 6e7
        qg = (mon["Q_total"][i] - mon["Q_liquid"][i]) * 6e4
        pb = mon["p5"][i]
        head = (f"{label} drain, pool {st['h']:g} mm, outlet end {rk.TILT:g}° up — t = {t * 1e3:4.1f} ms of flow\n"
                f"liquid out {ql:5.0f} mL/min   ·   air out {qg:4.1f} L/min   ·   pressure at the bore entrance "
                f"{pb:+5.0f} Pa   (model, settled: {st['qL_ml_min']:.0f} mL/min, {st['qG_l_min']:.1f} L/min)")
        for key, crop, stream in (("midplane", ((-8, 140), (-3.5, 36)), False),
                                  ("outlet_end", ((92, 136), (12, 34)), True)):
            p = os.path.join(out, f"{key}_{t * 1e3:05.1f}ms.png")
            rk.plane_world(g, frame, st, (0, 1, 0), (0, 0, 0), head, p, crop, stream)
            gifs[key].append(p)
        print(f"  frame t = {t * 1e3:.1f} ms")
    for key, paths in gifs.items():
        imgs = [Image.open(p).convert("RGB") for p in paths]
        w = min(1400, min(im.width for im in imgs))
        imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
        h = min(im.height for im in imgs)
        imgs = [np.asarray(im)[:h] for im in imgs]
        durations = [a.seconds * 1000] * (len(imgs) - 1) + [a.seconds * 3000]   # hold the last frame
        path = os.path.join(run, "pictures", f"{key}_slow.gif")
        imageio.mimsave(path, imgs, duration=durations, loop=0)
        print("wrote", path)


if __name__ == "__main__":
    main()

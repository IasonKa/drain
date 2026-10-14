"""
Export the CAD of both designs (window parts in the hole frame + sponge organ in the drain's
global frame) to out/cad/<design>/.

    python scripts/build_cad.py [--design standoff] [--no-stl]
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge import config, cad
from drainsponge.geom import Window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default=None, choices=list(config.DESIGNS))
    ap.add_argument("--out", default="out/cad")
    ap.add_argument("--no-stl", action="store_true")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "hole_center.json")))
    for d in ([a.design] if a.design else list(config.DESIGNS)):
        t = time.time()
        w = Window.from_dict(config.load(design=d)["window"])
        od = os.path.join(a.out, d)
        vols = cad.write_window(w, od, stl=not a.no_stl)
        vo = cad.write_organ(w, meta, od, stl=not a.no_stl)
        L = w.half
        print(f"{d}: volumes (mm3) {json.dumps({k: round(v, 4) for k, v in vols.items()})}, organ {vo:.0f}; "
              f"analytic tissue {(2*L)**2*w.tissue_h:.3f}, channel {(2*L)**2*w.channel_d:.3f}  ({time.time()-t:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()

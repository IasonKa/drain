"""
Write RUN/replay/replay.feb: the whole converged interface-pressure history of a coupled run
as ONE FEBio model (all windows as steps, plot file written at the end of each window), so
FEBio Studio can show the full tissue history from a single .xplt.

    python scripts/make_replay.py RUN
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drainsponge.geom import Window, WindowMesh
from drainsponge.febio_model import FebioModel


def main():
    run = sys.argv[1]
    cfg = json.load(open(os.path.join(run, "config.json")))
    st = json.load(open(os.path.join(run, "state.json")))
    H = np.load(os.path.join(run, "interface_history.npz"))
    n = st["n_done"]
    w = Window.from_dict(cfg["window"])
    fl = WindowMesh(w, zones=("channel", "hole", "stand", "film"))
    tis = WindowMesh(w, zones=("tissue",))
    fe = FebioModel(cfg, tis, fl)
    if not np.array_equal(fe.patch_node, H["patch_node"]):
        raise RuntimeError("patch layout differs from the run - same config/package version needed")
    out = os.path.join(run, "replay")
    fe.write(os.path.join(out, "replay.feb"), H["times"][:n], H["P"][:n + 1], plot="PLOT_FINAL")
    print(f"wrote {out}/replay.feb with {n} windows (t_end = {H['times'][n-1]:.4g} s)")


if __name__ == "__main__":
    main()

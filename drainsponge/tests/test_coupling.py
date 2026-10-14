"""
End-to-end test of the coupling driver with mock solvers (no OpenFOAM/FEBio needed):
restart chaining, stop before the wall time + resume, restart failure -> rerun fallback,
snapshots. Run: python tests/test_coupling.py [workdir]
"""
import os, sys, json, shutil, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from drainsponge import config
from drainsponge.couple import Coupler, EXIT_DONE, EXIT_RESUBMIT


def make(out):
    cfg = config.load(design="standoff", resolution="test",
                      time={"n_windows": 4, "t_end_s": 1.0, "t_first_s": 1e-3},
                      coupling={"max_iter": 3, "min_iter": 2, "n_patches": 40},
                      febio={"exe": [sys.executable, os.path.join(HERE, "mock_febio.py")]},
                      openfoam={"n_procs": 1, "write_compression": True})
    c = Coupler(cfg, out)
    mock = os.path.join(HERE, "mock_openfoam.py").replace("\\", "/")
    py = sys.executable.replace("\\", "/")
    open(os.path.join(c.case, "Allrun"), "w", newline="\n").write(f'#!/bin/bash\ncd "$(dirname "$0")"\n"{py}" "{mock}" .\n')
    return c


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    # 1) stop after the first window, then resume
    out = os.path.join(base, "run_restart")
    shutil.rmtree(out, ignore_errors=True)
    c = make(out)
    code = c.run(deadline=time.time() + 0.1)
    assert code == EXIT_RESUBMIT and c.state["n_done"] == 1, (code, c.state["n_done"])
    c2 = make(out)                                   # resume (Allrun mock re-written, state kept)
    assert c2.state["n_done"] == 1
    code = c2.run()
    assert code == EXIT_DONE, code
    st = json.load(open(os.path.join(out, "state.json")))
    H = np.load(os.path.join(out, "interface_history.npz"))
    assert st["n_done"] == 4 and st["mode"] == "restart" and H["P"].shape[0] == 5, (st["n_done"], st["mode"], H["P"].shape)
    assert os.path.exists(os.path.join(out, "febio", "w004")) and any(
        f.endswith("restart.feb") for _, _, fs in os.walk(os.path.join(out, "febio", "w004")) for f in fs)
    for f in ("tissue.pvd", "fluid.pvd", "interface.pvd", "interface_004.vtu", "fluid_004.vtu", "tissue_004.vtu"):
        assert os.path.exists(os.path.join(out, "vtk", f)), f
    print("restart chain + resume: OK;  interface pressure (Pa) per window:",
          np.round(H["P"].min(1)).tolist(), " min film (um):", [round(r["min_film_um"], 1) for r in st["log"]])
    # 2) FEBio restart fails -> automatic fallback to re-running from t = 0
    out = os.path.join(base, "run_fallback")
    shutil.rmtree(out, ignore_errors=True)
    os.environ["MOCK_FEBIO_FAIL_RESTART"] = "1"
    c = make(out)
    assert c.run() == EXIT_DONE
    st = json.load(open(os.path.join(out, "state.json")))
    assert st["mode"] == "rerun" and st["n_done"] == 4
    os.environ.pop("MOCK_FEBIO_FAIL_RESTART")
    print("restart failure -> rerun fallback: OK")
    print("all coupling tests passed in", base)


if __name__ == "__main__":
    main()

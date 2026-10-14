"""
Per-window snapshots of a coupled run for ParaView (VTU files + .pvd time collections):

  RUN/vtk/tissue_NNN.vtu      deformed tissue, point data: displacement (mm), pore pressure (Pa)
  RUN/vtk/fluid_NNN.vtu       liquid (channel/hole/stand/film) at the deformed shape, cell data p (Pa), U (m/s)
  RUN/vtk/interface_NNN.vtu   tissue underside: film pressure (Pa), film thickness (um)
  RUN/vtk/tissue.pvd, fluid.pvd, interface.pvd   open these in ParaView and press play

Also usable after a run: python -m drainsponge.post RUN   (rebuilds the .pvd files)
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np


def _write_vtu(path, points, cells, point_data=None, cell_data=None, ctype="hexahedron"):
    import meshio
    meshio.write(path, meshio.Mesh(points, [(ctype, cells)],
                                   point_data=point_data or {},
                                   cell_data={k: [v] for k, v in (cell_data or {}).items()}),
                 binary=True)


def _pvd(folder, name):
    files = sorted(glob.glob(os.path.join(folder, f"{name}_*.vtu")))
    times = json.load(open(os.path.join(folder, "times.json"))) if os.path.exists(os.path.join(folder, "times.json")) else {}
    with open(os.path.join(folder, f"{name}.pvd"), "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n<Collection>\n')
        for p in files:
            n = os.path.basename(p).split("_")[-1].split(".")[0]
            t = times.get(str(int(n)), int(n))
            f.write(f'<DataSet timestep="{t}" file="{os.path.basename(p)}"/>\n')
        f.write("</Collection>\n</VTKFile>\n")


def write_window_snapshot(cp, n, wd, pq):
    """cp: Coupler after committing window n; wd: FEBio run dir of the accepted iteration."""
    from .febio_model import last_block_at
    from .foam_case import latest_time, read_internal_scalar, read_internal_vector
    out = os.path.join(cp.out, "vtk")
    os.makedirs(out, exist_ok=True)
    t_n = float(cp.times[n - 1])
    tj = os.path.join(out, "times.json")
    times = json.load(open(tj)) if os.path.exists(tj) else {}
    times[str(n)] = t_n
    json.dump(times, open(tj, "w"))
    # tissue
    cands = [wd] + sorted(glob.glob(os.path.join(cp.out, "febio", "w*", "it*")))
    best = None
    for d in cands:
        f = os.path.join(d, "tissue_state.txt")
        if os.path.exists(f) and (best is None or os.path.getmtime(f) > os.path.getmtime(best)):
            best = f
    if best:
        tb, ids, v = last_block_at(best, t_n)
        if abs(tb - t_n) <= 1e-6 * max(1, t_n):
            X = cp.tis.points.copy()
            U = np.zeros_like(X)
            pe = np.zeros(len(X))
            U[ids - 1] = v[:, :3]
            pe[ids - 1] = v[:, 3] * 1e6
            _write_vtu(os.path.join(out, f"tissue_{n:03d}.vtu"), X + U, cp.tis.cells,
                       point_data={"displacement_mm": U, "pore_pressure_Pa": pe})
    # liquid
    lt = latest_time(cp.case)
    if lt:
        cd = {}
        try:
            cd["p_Pa"] = np.asarray(read_internal_scalar(os.path.join(cp.case, lt, "p"))) * cp.cfg["fluid"]["rho_kg_m3"]
            cd["U_m_s"] = read_internal_vector(os.path.join(cp.case, lt, "U"))
        except Exception:
            pass
        cd = {k: v for k, v in cd.items() if np.ndim(v) > 0 and len(v) == len(cp.fl.cells)}
        _write_vtu(os.path.join(out, f"fluid_{n:03d}.vtu"), cp.fl.points, cp.fl.cells, cell_data=cd)
    # interface (tissue underside) as a quad surface
    plan = cp.fl.plan
    P3 = np.c_[plan, cp.fl.zt]
    _write_vtu(os.path.join(out, f"interface_{n:03d}.vtu"), P3, cp.fl.quads,
               point_data={"film_um": cp.fl.film_thickness() * 1e3,
                           "patch_pressure_Pa": cp.P[-1][cp.fe.patch_node]},
               cell_data={"film_pressure_Pa": pq}, ctype="quad")
    for name in ("tissue", "fluid", "interface"):
        _pvd(out, name)


if __name__ == "__main__":
    folder = os.path.join(sys.argv[1], "vtk")
    for name in ("tissue", "fluid", "interface"):
        _pvd(folder, name)
    print("pvd files written in", folder)

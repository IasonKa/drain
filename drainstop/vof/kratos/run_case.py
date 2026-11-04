"""Run a Kratos two-fluid drain case and log the outlet flows and the pressure inside the drain.

    python vof/kratos/run_case.py <run dir> [--threads 14] [--steps N] [--resume]

--resume continues from the last field output (written every 5 ms of flow): the outputs so far are
moved to <run>/vtk_saved/, the latest one's velocity, pressure and level set are loaded onto the same
mesh and the run restarts at that time (the first step after a resume is first-order in time). Use it
after a reboot or a stop. Without --resume, vtk_saved/ from an earlier run is removed.

Writes <run>/monitor.csv every step (appended on --resume):
  t, dt, Q_total (m3/s out of the tube), Q_liquid, p at probes in the drain interior and their
  level set (negative = liquid), and the solver's wall time per step.
Field output (VELOCITY, PRESSURE, DISTANCE) goes to <run>/vtk_output/ as set in ProjectParameters.json.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication  # noqa: F401
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis

# probe points (mm, drain frame): top of the drain interior along the length, and the bore entrance
PROBES = [(15.0, 0.0, 8.07), (40.0, 0.0, 8.07), (65.0, 0.0, 8.07), (90.0, 0.0, 8.07), (112.0, 0.0, 8.07),
          (117.1, 0.0, 6.335), (117.1, 0.0, 3.335), (126.0, 0.0, 4.835)]


def last_snapshot(run):
    """(time, path) of the latest field output, or None.

    Kratos' VTK output process deletes vtk_output/ when a run starts, so the outputs written so far are
    first moved to vtk_saved/ (kept for the pictures) and the snapshot is read from there."""
    import glob
    import re
    import shutil
    saved = os.path.join(run, "vtk_saved")
    os.makedirs(saved, exist_ok=True)
    for f in glob.glob(os.path.join(run, "vtk_output", "*.vtk")):
        shutil.move(f, os.path.join(saved, os.path.basename(f)))
    files = glob.glob(os.path.join(saved, "*.vtk"))
    if not files:
        return None
    timed = [(float(re.findall(r"_([0-9.eE+-]+)\.vtk$", os.path.basename(f))[0]), f) for f in files]
    return max(timed)


class DrainAnalysis(FluidDynamicsAnalysis):
    def __init__(self, model, parameters, run, max_steps=None, snapshot=None):
        super().__init__(model, parameters)
        self.run_dir = run
        self.max_steps = max_steps
        self.n_steps = 0
        self.snapshot = snapshot

    def ModifyAfterSolverInitialize(self):
        super().ModifyAfterSolverInitialize()
        if not self.snapshot:
            return
        import numpy as np
        import pyvista as pv
        t0, path = self.snapshot
        g = pv.read(path)
        mp = self.model.GetModelPart("FluidModelPart")
        nodes = list(mp.Nodes)
        xyz = np.array([[n.X, n.Y, n.Z] for n in nodes])
        pts = np.asarray(g.points)
        # same nodes in the same order; VTK stores single precision, so allow 1 um (cells are >= 0.1 mm)
        dmax = float(np.max(np.abs(pts - xyz))) if pts.shape == xyz.shape else float("inf")
        if dmax > 1e-6:
            raise RuntimeError(f"{path}: the snapshot's nodes do not match this mesh (max diff {dmax:.3g} m)")
        U = np.asarray(g.point_data["VELOCITY"], float)
        P = np.asarray(g.point_data["PRESSURE"], float)
        D = np.asarray(g.point_data["DISTANCE"], float)
        for i, n in enumerate(nodes):
            v = KM.Array3([U[i, 0], U[i, 1], U[i, 2]])
            for k in (0, 1, 2):                  # current and previous steps (BDF2 history)
                n.SetSolutionStepValue(KM.VELOCITY, k, v)
                n.SetSolutionStepValue(KM.PRESSURE, k, P[i])
                n.SetSolutionStepValue(KM.DISTANCE, k, D[i])
        solver = self._GetSolver()
        if hasattr(solver, "_SetNodalProperties"):
            solver._SetNodalProperties()         # density and viscosity from the loaded level set
        print(f"resumed from {os.path.basename(path)} at t = {t0:.5f} s", flush=True)

    def _CreateSolver(self):
        # Kratos 10.3.1's solver wrapper looks for 'navier_stokes_two_fluid_solver' (typo); the module
        # is 'navier_stokes_two_fluids_solver', so create the solver directly
        from KratosMultiphysics.FluidDynamicsApplication import navier_stokes_two_fluids_solver
        return navier_stokes_two_fluids_solver.CreateSolver(self.model, self.project_parameters["solver_settings"])

    def Initialize(self):
        super().Initialize()
        mp = self.model.GetModelPart("FluidModelPart")
        self.outlet = mp.GetSubModelPart("Outlet")
        nodes = list(mp.Nodes)
        import numpy as np
        xyz = np.array([[n.X, n.Y, n.Z] for n in nodes])
        self.probe_nodes = []
        for p in PROBES:
            d = np.linalg.norm(xyz - np.array(p) * 1e-3, axis=1)
            self.probe_nodes.append(nodes[int(np.argmin(d))])
        mon = os.path.join(self.run_dir, "monitor.csv")
        if self.snapshot and os.path.exists(mon):  # keep the rows up to the snapshot time
            t0 = self.snapshot[0]
            with open(mon) as f:
                rows = [r for r in csv.reader(f)]
            keep = [rows[0]] + [r for r in rows[1:] if r and float(r[0]) <= t0 + 1e-12]
            self.csv = open(mon, "w", newline="")
            self.writer = csv.writer(self.csv)
            self.writer.writerows(keep)
        else:
            self.csv = open(mon, "w", newline="")
            self.writer = csv.writer(self.csv)
            self.writer.writerow(["t", "dt", "Q_total", "Q_liquid", "wall_s"]
                                 + [f"p{i}" for i in range(len(PROBES))] + [f"d{i}" for i in range(len(PROBES))])
        self.t_wall = time.time()

    def _outlet_flows(self):
        q_tot = q_liq = 0.0
        for c in self.outlet.Conditions:
            g = c.GetGeometry()
            p0, p1, p2 = (KM.Array3([n.X, n.Y, n.Z]) for n in g)
            e1 = [p1[i] - p0[i] for i in range(3)]
            e2 = [p2[i] - p0[i] for i in range(3)]
            nx = e1[1] * e2[2] - e1[2] * e2[1]
            ny = e1[2] * e2[0] - e1[0] * e2[2]
            nz = e1[0] * e2[1] - e1[1] * e2[0]
            vx = vy = vz = 0.0
            liq = 0.0
            for n in g:
                v = n.GetSolutionStepValue(KM.VELOCITY)
                vx += v[0]; vy += v[1]; vz += v[2]
                liq += 1.0 if n.GetSolutionStepValue(KM.DISTANCE) <= 0.0 else 0.0
            q = 0.5 * (vx * nx + vy * ny + vz * nz) / 3.0       # area-weighted normal flux (outward normal)
            q_tot += q
            q_liq += q * liq / 3.0
        return q_tot, q_liq

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        self.n_steps += 1
        mp = self.model.GetModelPart("FluidModelPart")
        t = mp.ProcessInfo[KM.TIME]
        dt = mp.ProcessInfo[KM.DELTA_TIME]
        q_tot, q_liq = self._outlet_flows()
        now = time.time()
        row = [t, dt, q_tot, q_liq, now - self.t_wall]
        row += [n.GetSolutionStepValue(KM.PRESSURE) for n in self.probe_nodes]
        row += [n.GetSolutionStepValue(KM.DISTANCE) for n in self.probe_nodes]
        self.t_wall = now
        self.writer.writerow([f"{v:.6g}" for v in row])
        self.csv.flush()
        if self.n_steps % 10 == 0:
            print(f"step {self.n_steps}: t = {t:.5f} s, dt = {dt:.2e} s, liquid out {q_liq * 6e7:7.1f} mL/min, "
                  f"air out {(q_tot - q_liq) * 6e4:6.2f} L/min, {row[4]:.1f} s/step", flush=True)

    def KeepAdvancingSolutionLoop(self):
        if self.max_steps is not None and self.n_steps >= self.max_steps:
            return False
        return super().KeepAdvancingSolutionLoop()

    def Finalize(self):
        super().Finalize()
        self.csv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=None, help="stop after this many steps (test)")
    ap.add_argument("--resume", action="store_true", help="continue from the last field output")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    os.chdir(run)
    if sys.platform == "win32":                  # no idle sleep while this process runs (a run takes hours)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    KM.ParallelUtilities.SetNumThreads(a.threads)
    KM.Logger.GetDefaultOutput().SetSeverity(KM.Logger.Severity.WARNING if not os.environ.get("KRATOS_INFO") else KM.Logger.Severity.INFO)
    with open("ProjectParameters.json") as f:
        params = KM.Parameters(f.read())
    snapshot = last_snapshot(run) if a.resume else None
    if snapshot:
        params["problem_data"]["start_time"].SetDouble(snapshot[0])
    elif a.resume:
        print("nothing to resume from: starting at t = 0", flush=True)
    else:
        import shutil
        shutil.rmtree(os.path.join(run, "vtk_saved"), ignore_errors=True)
        for f in ("FINISHED",):
            if os.path.exists(f):
                os.remove(f)
    model = KM.Model()
    sim = DrainAnalysis(model, params, run, a.steps, snapshot)
    t0 = time.time()
    sim.Run()
    print(f"done in {time.time() - t0:.0f} s, {sim.n_steps} steps")
    if a.steps is None:
        open("FINISHED", "w").write(f"{sim.n_steps} steps, {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()

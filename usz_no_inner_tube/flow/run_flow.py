"""
Run a flow case to its stationary state (implicit time marching with growing steps).
  <kratos python> run_flow.py runs/<case> [--threads 12] [--steps 60]
Writes result.npz (nodal velocity m/s and pressure Pa) and history.csv (flows per step, mL/min).
"""
import argparse, os, time
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication  # noqa
from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis


class Run(FluidDynamicsAnalysis):
    def __init__(self, model, p, a):
        super().__init__(model, p)
        self.a = a
        m = np.load("mesh.npz")
        x = m["xyz"] * 1e-3
        t = m["tris"]
        self.tris, self.lab = t, m["lab"]
        self.An = 0.5 * np.cross(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])     # outward area vectors
        self.prev = None
        self.k = 0
        self.done = False
        self.t0 = time.time()
        self.dts = np.minimum(a.dt0 * a.grow ** np.arange(a.steps), a.dtmax)
        nh = max(int(self.lab.max()) - 2, 0)
        open("history.csv", "w").write("step,time,dt,change,inlet,outlet" + "".join(f",hole{i}" for i in range(nh)) + "\n")

    def _fields(self):
        mp = self._GetSolver().GetComputingModelPart()
        n = mp.NumberOfNodes()
        v = np.empty((n, 3))
        p = np.empty(n)
        for i, nd in enumerate(mp.Nodes):
            v[i, 0] = nd.GetSolutionStepValue(KM.VELOCITY_X)
            v[i, 1] = nd.GetSolutionStepValue(KM.VELOCITY_Y)
            v[i, 2] = nd.GetSolutionStepValue(KM.VELOCITY_Z)
            p[i] = nd.GetSolutionStepValue(KM.PRESSURE)
        return v, p

    def RunSolutionLoop(self):
        s = self._GetSolver()
        while self.KeepAdvancingSolutionLoop():
            s.settings["time_stepping"]["time_step"].SetDouble(float(self.dts[min(self.k, len(self.dts) - 1)]))
            self.time = s.AdvanceInTime(self.time)
            self.InitializeSolutionStep()
            s.Predict()
            s.SolveSolutionStep()
            self.FinalizeSolutionStep()
            self.OutputSolutionStep()

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        v, p = self._fields()
        q = np.einsum("ij,ij->i", v[self.tris].mean(axis=1), self.An) * 6e7              # mL/min, + = out of the domain
        fl = [q[self.lab == i].sum() for i in range(1, max(int(self.lab.max()), 2) + 1)]
        ch = np.nan if self.prev is None else float(np.abs(v - self.prev).max() / max(np.abs(v).max(), 1e-12))
        self.prev = v.copy()
        self.k += 1
        dt = self.dts[min(self.k - 1, len(self.dts) - 1)]
        open("history.csv", "a").write(f"{self.k},{self.time:.5g},{dt:.4g},{ch:.4g}," + ",".join(f"{x:.5g}" for x in fl) + "\n")
        print(f"step {self.k} t={self.time:.4g}s dt={dt:.3g} change={ch:.3g} vmax={np.abs(v).max():.3g} "
              f"flows(mL/min,+out)={np.round(fl, 3)} wall {time.time() - self.t0:.0f}s", flush=True)
        if not np.isfinite(v).all() or np.abs(v).max() > 100:
            print("DIVERGED", flush=True)
            self.done = True
            return
        if self.k % 5 == 0:
            np.savez("result.npz", v=v, p=p, step=self.k, time=self.time)
        if ch == ch and ch < self.a.tol and self.k > 5:
            print("stationary", flush=True)
            np.savez("result.npz", v=v, p=p, step=self.k, time=self.time)
            self.done = True

    def KeepAdvancingSolutionLoop(self):
        return (not self.done) and self.k < self.a.steps

    def Finalize(self):
        v, p = self._fields()
        if np.isfinite(v).all() and np.abs(v).max() < 100:
            np.savez("result.npz", v=v, p=p, step=self.k, time=self.time)
        super().Finalize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--dt0", type=float, default=0.002)
    ap.add_argument("--grow", type=float, default=1.25)
    ap.add_argument("--dtmax", type=float, default=2.0)
    a = ap.parse_args()
    os.chdir(a.run)
    KM.ParallelUtilities.SetNumThreads(a.threads)
    Run(KM.Model(), KM.Parameters(open("ProjectParameters.json").read()), a).Run()

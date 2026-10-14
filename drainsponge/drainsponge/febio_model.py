"""
FEBio 4 model of the sponge tissue on the drain window (biphasic, transient), written
for coupling window by window.

  tissue  : biphasic = neo-Hookean solid matrix + interstitial liquid, permeability
            constant or Holmes-Mow; hex8, same nodes as the OpenFOAM film top (1:1)
  drain   : rigid body of quad4 shells = outer skin + hole wall (+ standoff ring), fixed
  contact : sliding-elastic tissue bottom -> drain (friction), as in drainfeb's pilot
  top     : held by the organ (fixed), liquid at body pressure (p = 0)
  sides   : symmetry (normal displacement 0); drained (p = 0) only when the film edge is
            open to free liquid ("open" scenario)
  bottom  : the liquid film from OpenFOAM acts on it, per interface PATCH j:
              fluid pressure  p = P_j(t)   (prescribed fluid pressure, load curve j)
              total traction  = -P_j(t) n  (pressure load, same load curve j)

Time: one FEBio step per coupling window. The interface pressure history of every patch
is a load curve (piecewise linear in time through the window ends t_n).

Two ways to run window n (FEBio 4.13 restart semantics, see FEBioXML/FERestartImport.cpp):
  restart : febio4 -r restart_n.feb -dump=state.dmp
            restart_n.feb = <febio_restart version="4.0"> Archive = dump at t_{n-1},
            LoadData redefines all patch curves (they are referenced by index, so the BCs
            follow), Step adds one new step for [t_{n-1}, t_n]. Cost per window is constant.
  rerun   : a full model with steps 1..n from t = 0 (fallback; cost grows with n)
Logs are written once at the end of every step (output_level OUTPUT_FINAL).
Units: mm, N, s, MPa  (k [mm^4/(N s)] = k [m^4/(N s)] * 1e12).
"""
from __future__ import annotations
import os, json
import numpy as np

from .geom import WindowMesh


def _ids(a):
    return ",".join(str(int(i)) for i in a)


def make_patches(plan, quads, K, seed=1):
    """Group the interface into K patches (k-means in r^0.6 - stretched polar coordinates,
    so patches are smaller near the hole). Returns patch id per plan node and per plan quad."""
    from scipy.cluster.vq import kmeans2
    r = np.hypot(plan[:, 0], plan[:, 1])
    th = np.arctan2(plan[:, 1], plan[:, 0])
    s = r ** 0.6
    X = np.c_[s * np.cos(th), s * np.sin(th)]
    K = int(min(K, len(plan)))
    rng = np.random.default_rng(seed)
    cen, _ = kmeans2(X, X[rng.choice(len(X), K, replace=False)], iter=30, minit="matrix")

    def nearest(Y):
        d = ((Y[:, None, :] - cen[None]) ** 2).sum(-1)
        return np.argmin(d, 1)
    pn = nearest(X)
    qc = plan[quads].mean(1)
    rq, tq = np.hypot(qc[:, 0], qc[:, 1]), np.arctan2(qc[:, 1], qc[:, 0])
    pq = nearest(np.c_[rq ** 0.6 * np.cos(tq), rq ** 0.6 * np.sin(tq)])
    used = np.unique(np.r_[pn, pq])
    remap = -np.ones(K, int)
    remap[used] = np.arange(len(used))
    return remap[pn], remap[pq], len(used)


class FebioModel:
    def __init__(self, cfg, wm_tissue: WindowMesh, wm_fluid: WindowMesh):
        self.cfg = cfg
        self.t = wm_tissue
        w = wm_tissue.w
        P = wm_tissue.points
        self.nodes = P
        self.hex = wm_tissue.cells
        b = ~wm_tissue.f_internal
        fp = wm_tissue.f_patch
        bf = np.nonzero(b & (fp == "tissue_bottom"))[0]
        self.bottom = wm_tissue.f_nodes[bf]
        self.bottom_quad = wm_tissue.cell_plan[wm_tissue.f_owner[bf]]          # plan quad of each bottom face
        self.bottom_elem = wm_tissue.f_owner[bf]
        self.top_nodes = np.unique(wm_tissue.f_nodes[b & (fp == "tissue_top")])
        side = np.unique(wm_tissue.f_nodes[b & (fp == "tissue_side")])
        self.side_x = side[np.abs(np.abs(P[side, 0]) - w.half) < 1e-6]
        self.side_y = side[np.abs(np.abs(P[side, 1]) - w.half) < 1e-6]
        self.iface_nodes = wm_tissue.iface_node_ids()                          # plan order
        self.bottom_set = np.unique(self.bottom)
        self.plan_of_bottom = wm_tissue.node_plan[self.bottom_set]
        # rigid drain: skin + hole wall + standoff faces of the fluid mesh, normals towards the tissue
        bfl = ~wm_fluid.f_internal
        sel = bfl & np.isin(wm_fluid.f_patch, ["drain_skin", "hole_wall", "standoff"])
        q = wm_fluid.f_nodes[sel][:, ::-1]
        used, inv = np.unique(q, return_inverse=True)
        self.drain_nodes = wm_fluid.points[used]
        self.drain_quads = inv.reshape(-1, 4)
        # interface patches
        K = cfg["coupling"].get("n_patches", 600)
        self.patch_node, self.patch_quad, self.K = make_patches(wm_tissue.plan, wm_tissue.quads, K)
        pq = self.patch_quad[self.bottom_quad]
        pn = self.patch_node[self.plan_of_bottom]
        self.patch_faces = [np.nonzero(pq == j)[0] for j in range(self.K)]
        self.patch_nodes = [self.bottom_set[pn == j] for j in range(self.K)]
        # plan-quad areas for area-weighted patch averages
        pl = wm_tissue.plan[wm_tissue.quads]
        self.quad_area = 0.5 * np.abs((pl[:, 2, 0] - pl[:, 0, 0]) * (pl[:, 3, 1] - pl[:, 1, 1])
                                      - (pl[:, 3, 0] - pl[:, 1, 0]) * (pl[:, 2, 1] - pl[:, 0, 1]))
        self.open_edge = cfg.get("boundary", {}).get("film_edge", "open") == "open"

    # ------------------------------------------------------------------ patch helpers
    def patch_average(self, value_per_quad):
        """Area-weighted average of a per-plan-quad field over every patch."""
        A = self.quad_area
        num = np.bincount(self.patch_quad, A * value_per_quad, minlength=self.K)
        den = np.bincount(self.patch_quad, A, minlength=self.K)
        out = num / np.maximum(den, 1e-30)
        empty = den == 0                                   # node-only patches: use the overall mean
        if empty.any():
            out[empty] = np.average(value_per_quad, weights=A)
        return out

    def patch_to_plan(self, P_patch):
        return P_patch[self.patch_node]

    # ------------------------------------------------------------------ XML pieces
    def _control(self, dt_window, nst, plot):
        return f"""<Control>
\t\t\t\t<analysis>TRANSIENT</analysis>
\t\t\t\t<time_steps>{nst}</time_steps>
\t\t\t\t<step_size>{dt_window/nst:.8g}</step_size>
\t\t\t\t<plot_level>{plot}</plot_level>
\t\t\t\t<output_level>OUTPUT_FINAL</output_level>
\t\t\t\t<time_stepper type="default">
\t\t\t\t\t<max_retries>12</max_retries>
\t\t\t\t\t<opt_iter>12</opt_iter>
\t\t\t\t\t<dtmin>{dt_window*1e-6:.8g}</dtmin>
\t\t\t\t\t<dtmax>{dt_window/nst:.8g}</dtmax>
\t\t\t\t\t<aggressiveness>0</aggressiveness>
\t\t\t\t\t<cutback>0.5</cutback>
\t\t\t\t</time_stepper>
\t\t\t\t<solver type="biphasic">
\t\t\t\t\t<symmetric_stiffness>non-symmetric</symmetric_stiffness>
\t\t\t\t\t<max_refs>30</max_refs>
\t\t\t\t\t<diverge_reform>1</diverge_reform>
\t\t\t\t\t<reform_each_time_step>1</reform_each_time_step>
\t\t\t\t\t<dtol>0.001</dtol>
\t\t\t\t\t<etol>0.01</etol>
\t\t\t\t\t<rtol>0</rtol>
\t\t\t\t\t<ptol>0.01</ptol>
\t\t\t\t\t<lstol>0.9</lstol>
\t\t\t\t\t<ls_check_jacobians>1</ls_check_jacobians>
\t\t\t\t\t<qn_method type="BROYDEN"/>
\t\t\t\t</solver>
\t\t\t</Control>"""

    def _loadcurves(self, times, P_hist):
        """One load curve per patch: points (t_i, P_i [MPa]) for i = 0..n (t_0 = 0)."""
        t = np.r_[0.0, np.asarray(times, float)]
        Pm = np.asarray(P_hist, float) * 1e-6
        X = []
        for j in range(self.K):
            X.append(f'\t\t<load_controller id="{j+1}" name="patch{j+1}" type="loadcurve">\n'
                     '\t\t\t<interpolate>LINEAR</interpolate>\n\t\t\t<extend>CONSTANT</extend>\n\t\t\t<points>\n')
            X += [f"\t\t\t\t<pt>{a:.10g},{v:.8g}</pt>\n" for a, v in zip(t, Pm[:, j])]
            X.append("\t\t\t</points>\n\t\t</load_controller>\n")
        return X

    def _step(self, n, t0, t1, plot):
        nst = int(self.cfg["time"].get("febio_steps_per_window", 6))
        return (f'\t\t<step id="{n}" name="window{n}">\n\t\t\t' + self._control(t1 - t0, nst, plot) + "\n\t\t</step>\n")

    # ------------------------------------------------------------------ full model
    def write(self, path, times, P_hist_patch, plot="PLOT_FINAL"):
        """Full model with steps for windows 1..n. times: t_1..t_n (s);
        P_hist_patch: (n+1, K) interface pressure per patch in Pa, row 0 = t = 0."""
        cfg = self.cfg
        T = cfg["tissue"]
        E = T["E_Pa"] * 1e-6
        k = T["perm_m4_Ns"] * 1e12
        times = np.asarray(times, float)
        N = len(times)
        nT = len(self.nodes)
        X = ['<?xml version="1.0" encoding="ISO-8859-1"?>\n<febio_spec version="4.0">\n',
             '\t<Module type="biphasic"/>\n']
        if T.get("perm_M", 0) > 0 or T.get("perm_alpha", 0) > 0:
            perm = (f'\t\t\t<permeability type="perm-Holmes-Mow">\n\t\t\t\t<perm>{k:.6g}</perm>\n'
                    f'\t\t\t\t<M>{T["perm_M"]}</M>\n\t\t\t\t<alpha>{T["perm_alpha"]}</alpha>\n\t\t\t</permeability>\n')
        else:
            perm = f'\t\t\t<permeability type="perm-const-iso">\n\t\t\t\t<perm>{k:.6g}</perm>\n\t\t\t</permeability>\n'
        X.append(f"""\t<Globals>
\t\t<Constants>
\t\t\t<T>0</T>
\t\t\t<R>0</R>
\t\t\t<Fc>0</Fc>
\t\t</Constants>
\t</Globals>
\t<Material>
\t\t<material id="1" name="tissue" type="biphasic">
\t\t\t<phi0>{T['phi0']}</phi0>
\t\t\t<fluid_density>1e-9</fluid_density>
\t\t\t<solid type="neo-Hookean">
\t\t\t\t<density>1e-9</density>
\t\t\t\t<E>{E:.6g}</E>
\t\t\t\t<v>{T['nu']}</v>
\t\t\t</solid>
{perm}\t\t</material>
\t\t<material id="2" name="drain" type="rigid body">
\t\t\t<density>1e-9</density>
\t\t\t<E>1</E>
\t\t\t<v>0.3</v>
\t\t</material>
\t</Material>
\t<Mesh>
\t\t<Nodes name="TissueNodes">
""")
        X += [f'\t\t\t<node id="{i+1}">{p[0]:.7g},{p[1]:.7g},{p[2]:.7g}</node>\n' for i, p in enumerate(self.nodes)]
        X.append('\t\t</Nodes>\n\t\t<Nodes name="DrainNodes">\n')
        X += [f'\t\t\t<node id="{nT+i+1}">{p[0]:.7g},{p[1]:.7g},{p[2]:.7g}</node>\n' for i, p in enumerate(self.drain_nodes)]
        X.append('\t\t</Nodes>\n\t\t<Elements type="hex8" name="Tissue">\n')
        X += [f'\t\t\t<elem id="{i+1}">{_ids(c + 1)}</elem>\n' for i, c in enumerate(self.hex)]
        nE = len(self.hex)
        X.append('\t\t</Elements>\n\t\t<Elements type="quad4" name="Drain">\n')
        X += [f'\t\t\t<elem id="{nE+i+1}">{_ids(q + nT + 1)}</elem>\n' for i, q in enumerate(self.drain_quads)]
        X.append('\t\t</Elements>\n')
        X.append('\t\t<Surface name="TissueBottom">\n')
        X += [f'\t\t\t<quad4 id="{i+1}">{_ids(f + 1)}</quad4>\n' for i, f in enumerate(self.bottom)]
        X.append('\t\t</Surface>\n\t\t<Surface name="DrainSurface">\n')
        X += [f'\t\t\t<quad4 id="{i+1}">{_ids(q + nT + 1)}</quad4>\n' for i, q in enumerate(self.drain_quads)]
        X.append('\t\t</Surface>\n')
        for j in range(self.K):
            fj = self.patch_faces[j]
            if len(fj):
                X.append(f'\t\t<Surface name="ps{j+1}">\n')
                X += [f'\t\t\t<quad4 id="{i+1}">{_ids(self.bottom[f] + 1)}</quad4>\n' for i, f in enumerate(fj)]
                X.append('\t\t</Surface>\n')
        notb = lambda a: np.setdiff1d(a, np.union1d(self.bottom_set, self.top_nodes))
        sets = [("Top", self.top_nodes), ("SideX", self.side_x), ("SideY", self.side_y),
                ("SideXp", notb(self.side_x)), ("SideYp", notb(self.side_y)), ("Bottom", self.bottom_set),
                ("AllTissue", np.arange(nT))]
        sets += [(f"pn{j+1}", self.patch_nodes[j]) for j in range(self.K) if len(self.patch_nodes[j])]
        for name, ns in sets:
            X.append(f'\t\t<NodeSet name="{name}">{_ids(ns + 1)}</NodeSet>\n')
        X.append(f'\t\t<ElementSet name="BottomElems">{_ids(np.unique(self.bottom_elem) + 1)}</ElementSet>\n')
        X.append('\t\t<SurfacePair name="TissueOnDrain">\n\t\t\t<primary>TissueBottom</primary>\n'
                 '\t\t\t<secondary>DrainSurface</secondary>\n\t\t</SurfacePair>\n')
        X.append('\t</Mesh>\n\t<MeshDomains>\n\t\t<SolidDomain name="Tissue" mat="tissue"/>\n'
                 '\t\t<ShellDomain name="Drain" mat="drain">\n\t\t\t<shell_thickness>0.05</shell_thickness>\n'
                 '\t\t</ShellDomain>\n\t</MeshDomains>\n')
        # ---- boundary conditions (global: active in every step, also in restart steps)
        X.append("""\t<Boundary>
\t\t<bc name="top_held" node_set="Top" type="zero displacement">
\t\t\t<x_dof>1</x_dof>
\t\t\t<y_dof>1</y_dof>
\t\t\t<z_dof>1</z_dof>
\t\t</bc>
\t\t<bc name="top_drained" node_set="Top" type="zero fluid pressure"/>
\t\t<bc name="side_x" node_set="SideX" type="zero displacement">
\t\t\t<x_dof>1</x_dof>
\t\t\t<y_dof>0</y_dof>
\t\t\t<z_dof>0</z_dof>
\t\t</bc>
\t\t<bc name="side_y" node_set="SideY" type="zero displacement">
\t\t\t<x_dof>0</x_dof>
\t\t\t<y_dof>1</y_dof>
\t\t\t<z_dof>0</z_dof>
\t\t</bc>
""")
        if self.open_edge:
            X.append('\t\t<bc name="sides_drained_x" node_set="SideXp" type="zero fluid pressure"/>\n'
                     '\t\t<bc name="sides_drained_y" node_set="SideYp" type="zero fluid pressure"/>\n')
        for j in range(self.K):
            if len(self.patch_nodes[j]):
                X.append(f'\t\t<bc name="film_p{j+1}" node_set="pn{j+1}" type="prescribed fluid pressure">\n'
                         f'\t\t\t<value lc="{j+1}">1</value>\n\t\t\t<relative>0</relative>\n\t\t</bc>\n')
        X.append("\t</Boundary>\n")
        X.append("""\t<Rigid>
\t\t<rigid_bc name="drain_fixed" type="rigid_fixed">
\t\t\t<rb>2</rb>
\t\t\t<Rx_dof>1</Rx_dof>
\t\t\t<Ry_dof>1</Ry_dof>
\t\t\t<Rz_dof>1</Rz_dof>
\t\t\t<Ru_dof>1</Ru_dof>
\t\t\t<Rv_dof>1</Rv_dof>
\t\t\t<Rw_dof>1</Rw_dof>
\t\t</rigid_bc>
\t</Rigid>
\t<Loads>
""")
        for j in range(self.K):
            if len(self.patch_faces[j]):
                X.append(f'\t\t<surface_load name="film_load{j+1}" surface="ps{j+1}" type="pressure">\n'
                         f'\t\t\t<pressure lc="{j+1}">1</pressure>\n\t\t\t<symmetric_stiffness>0</symmetric_stiffness>\n'
                         '\t\t\t<linear>0</linear>\n\t\t</surface_load>\n')
        X.append("\t</Loads>\n")
        mu_f = T.get("friction", 0.2)
        X.append(f"""\t<Contact>
\t\t<contact name="tissue_on_drain" surface_pair="TissueOnDrain" type="sliding-elastic">
\t\t\t<laugon>PENALTY</laugon>
\t\t\t<tolerance>0.2</tolerance>
\t\t\t<gaptol>0</gaptol>
\t\t\t<penalty>1</penalty>
\t\t\t<auto_penalty>1</auto_penalty>
\t\t\t<update_penalty>1</update_penalty>
\t\t\t<two_pass>0</two_pass>
\t\t\t<search_tol>0.01</search_tol>
\t\t\t<symmetric_stiffness>0</symmetric_stiffness>
\t\t\t<search_radius>0.2</search_radius>
\t\t\t<seg_up>0</seg_up>
\t\t\t<tension>0</tension>
\t\t\t<fric_coeff>{mu_f}</fric_coeff>
\t\t\t<slip_tol>0.05</slip_tol>
\t\t\t<smooth_aug>0</smooth_aug>
\t\t</contact>
\t</Contact>
""")
        X.append("\t<Step>\n")
        t_prev = 0.0
        for n in range(1, N + 1):
            X.append(self._step(n, t_prev, times[n - 1], plot))
            t_prev = times[n - 1]
        X.append("\t</Step>\n\t<LoadData>\n")
        X += self._loadcurves(times, P_hist_patch)
        X.append("\t</LoadData>\n")
        X.append("""\t<Output>
\t\t<plotfile type="febio">
\t\t\t<var type="displacement"/>
\t\t\t<var type="effective fluid pressure"/>
\t\t\t<var type="fluid flux"/>
\t\t\t<var type="stress"/>
\t\t\t<var type="Lagrange strain"/>
\t\t\t<var type="relative volume"/>
\t\t\t<var type="contact pressure"/>
\t\t\t<var type="contact gap"/>
\t\t</plotfile>
\t\t<logfile>
\t\t\t<node_data file="bottom_xyz.txt" data="x;y;z" delim="," node_set="Bottom"/>
\t\t\t<element_data file="bottom_flux.txt" data="wx;wy;wz" delim="," elem_set="BottomElems"/>
\t\t\t<node_data file="tissue_state.txt" data="ux;uy;uz;pe" delim="," node_set="AllTissue"/>
\t\t\t<rigid_body_data file="drain_force.txt" data="Fx;Fy;Fz" delim=",">2</rigid_body_data>
\t\t</logfile>
\t</Output>
</febio_spec>
""")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as f:
            f.write("".join(X))
        json.dump(dict(times=times.tolist(), n_tissue_nodes=int(nT), n_hex=int(nE), n_patches=int(self.K),
                       n_drain_quads=int(len(self.drain_quads)), n_bottom_faces=int(len(self.bottom))),
                  open(os.path.join(os.path.dirname(os.path.abspath(path)), "febio_info.json"), "w"), indent=1)
        return path

    # ------------------------------------------------------------------ restart file
    def write_restart(self, path, dump, times, P_hist_patch, plot="PLOT_NEVER"):
        """Restart input for window n = len(times): continue from `dump` (state at t_{n-1}),
        redefine every patch load curve with the history through t_n, add one step."""
        times = np.asarray(times, float)
        n = len(times)
        t0 = times[n - 2] if n > 1 else 0.0
        X = ['<?xml version="1.0" encoding="ISO-8859-1"?>\n<febio_restart version="4.0">\n',
             f"\t<Archive>{os.path.abspath(dump)}</Archive>\n", "\t<LoadData>\n"]
        X += self._loadcurves(times, P_hist_patch)
        X.append("\t</LoadData>\n\t<Step>\n")
        X.append(self._step(n, t0, times[n - 1], plot))
        X.append("\t</Step>\n</febio_restart>\n")
        with open(path, "w", newline="\n") as f:
            f.write("".join(X))
        return path


# ------------------------------------------------------------------ reading results
def read_log_blocks(path):
    """All data blocks of an FEBio logfile -> list of (time, ids, values)."""
    out, cur, t = [], None, None
    for line in open(path):
        s = line.strip()
        if s.startswith("*Time"):
            t = float(s.split("=")[1])
        elif s.startswith("*Data"):
            cur = ([], [])
            out.append((t, cur))
        elif s and not s.startswith("*") and cur is not None:
            v = s.replace(",", " ").split()
            cur[0].append(int(v[0]))
            cur[1].append([float(x) for x in v[1:]])
    return [(t, np.array(i), np.array(v)) for t, (i, v) in out]


def last_block_at(path, t, rtol=1e-6):
    """The LAST data block written at time t (logs may be appended across restarts/re-runs).
    Scans the file backwards, so it stays fast when the log grows to GBs."""
    import mmap
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size == 0:
            raise RuntimeError(f"no data in {path}")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        end = size
        best = None
        try:
            while True:
                i = mm.rfind(b"*Time", 0, end)
                if i < 0:
                    break
                eol = mm.find(b"\n", i)
                tb = float(mm[i:eol].split(b"=")[1])
                if best is None or abs(tb - t) < abs(best[0] - t):
                    best = (tb, i)
                if abs(tb - t) <= rtol * max(1.0, abs(t)):
                    best = (tb, i)
                    break
                end = i
            if best is None:
                raise RuntimeError(f"no data in {path}")
            tb, i = best
            j = mm.find(b"*Data", i)
            k = mm.find(b"\n", j) + 1
            nxt = mm.find(b"*Step", k)
            blk = mm[k: nxt if nxt >= 0 else size].decode()
        finally:
            mm.close()
    rows = [l.replace(",", " ").split() for l in blk.splitlines() if l.strip() and not l.startswith("*")]
    a = np.array(rows, float)
    return tb, a[:, 0].astype(int), a[:, 1:]

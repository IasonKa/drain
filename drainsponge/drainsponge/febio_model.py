"""
FEBio 4 model of the sponge tissue on the drain window (biphasic, transient).

  tissue  : biphasic = neo-Hookean solid matrix + interstitial liquid, permeability
            constant or Holmes-Mow (strain dependent: pores close when compressed)
            hex8, same nodes as the OpenFOAM film top (1:1 interface)
  drain   : rigid body of quad4 shells = outer skin (+ fillet) + hole wall, fixed
  contact : sliding-elastic tissue bottom -> drain (friction), as in drainfeb's pilot
  top     : held by the organ (fixed), liquid at body pressure (p = 0)
  sides   : symmetry (normal displacement 0), p = 0 (tissue continues outwards)
  bottom  : the liquid film from OpenFOAM acts on it:
              fluid pressure p = p_film(x, t)            (pressure continuity)
              total traction  = -p_film n  (pressure load, pulls when p_film < 0)
            p_film(x, t) is given at every interface node at the coupling times
            t_1 .. t_N and interpolated linearly in time: one FEBio step per coupling
            window, with a RELATIVE prescribed pressure map (P_n - P_n-1) ramped over
            the step, and two pressure-load maps (P_n-1 fading out, P_n fading in).
Units: mm, N, s, MPa  (k [mm^4/(N s)] = k [m^4/(N s)] * 1e12).
"""
from __future__ import annotations
import os, json
import numpy as np

from .geom import WindowMesh


def _ids(a):
    return ",".join(str(int(i)) for i in a)


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
        self.bottom = wm_tissue.f_nodes[b & (fp == "tissue_bottom")]
        self.top_nodes = np.unique(wm_tissue.f_nodes[b & (fp == "tissue_top")])
        side = np.unique(wm_tissue.f_nodes[b & (fp == "tissue_side")])
        self.side_x = side[np.abs(np.abs(P[side, 0]) - w.half) < 1e-6]
        self.side_y = side[np.abs(np.abs(P[side, 1]) - w.half) < 1e-6]
        self.iface_nodes = wm_tissue.iface_node_ids()               # plan order
        # element adjacent to each bottom face (for the flux out of the tissue)
        bf = np.nonzero(b & (fp == "tissue_bottom"))[0]
        self.bottom_elem = wm_tissue.f_owner[bf]
        # rigid drain: skin + hole wall faces of the fluid mesh, normals turned towards the tissue
        bfl = ~wm_fluid.f_internal
        sel = bfl & np.isin(wm_fluid.f_patch, ["drain_skin", "hole_wall"])
        q = wm_fluid.f_nodes[sel][:, ::-1]
        used, inv = np.unique(q, return_inverse=True)
        self.drain_nodes = wm_fluid.points[used]
        self.drain_quads = inv.reshape(-1, 4)
        # node lid of every interface node inside the bottom node set
        self.bottom_set = np.unique(self.bottom)
        lid = -np.ones(len(P), int)
        lid[self.bottom_set] = np.arange(len(self.bottom_set))
        self.lid = lid

    # ------------------------------------------------------------------ writer
    def write(self, path, times, P_hist_Pa, plot_every=1):
        """times: t_1..t_N (s); P_hist_Pa: (N+1, n_plan) interface pressure, row 0 = 0 at t=0."""
        cfg = self.cfg
        T = cfg["tissue"]
        E = T["E_Pa"] * 1e-6
        k = T["perm_m4_Ns"] * 1e12                                      # m^4/(N s) -> mm^4/(N s)
        times = np.asarray(times, float)
        N = len(times)
        Ph = np.asarray(P_hist_Pa, float) * 1e-6                       # MPa, plan order
        # plan order -> bottom node set order
        Pnode = np.zeros((N + 1, len(self.bottom_set)))
        Pnode[:, self.lid[self.iface_nodes]] = Ph
        # face values = mean of their 4 nodes
        Pface = Pnode[:, self.lid[self.bottom]].mean(2)
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
        # drained sets exclude the bottom edge (that edge gets the film pressure instead)
        notb = lambda a: np.setdiff1d(a, np.union1d(self.bottom_set, self.top_nodes))
        for name, ns in (("Top", self.top_nodes), ("SideX", self.side_x), ("SideY", self.side_y),
                         ("SideXp", notb(self.side_x)), ("SideYp", notb(self.side_y)),
                         ("Bottom", self.bottom_set)):
            X.append(f'\t\t<NodeSet name="{name}">{_ids(ns + 1)}</NodeSet>\n')
        X.append(f'\t\t<ElementSet name="BottomElems">{_ids(np.unique(self.bottom_elem) + 1)}</ElementSet>\n')
        X.append('\t\t<SurfacePair name="TissueOnDrain">\n\t\t\t<primary>TissueBottom</primary>\n'
                 '\t\t\t<secondary>DrainSurface</secondary>\n\t\t</SurfacePair>\n')
        X.append('\t</Mesh>\n\t<MeshDomains>\n\t\t<SolidDomain name="Tissue" mat="tissue"/>\n'
                 '\t\t<ShellDomain name="Drain" mat="drain">\n\t\t\t<shell_thickness>0.05</shell_thickness>\n'
                 '\t\t</ShellDomain>\n\t</MeshDomains>\n')
        # ---- maps: per window a relative node map and the face map at its end
        X.append('\t<MeshData>\n')
        for n in range(1, N + 1):
            d = Pnode[n] - Pnode[n - 1]
            X.append(f'\t\t<NodeData name="dP{n}" node_set="Bottom">\n')
            X += [f'\t\t\t<node lid="{i+1}">{v:.7g}</node>\n' for i, v in enumerate(d)]
            X.append('\t\t</NodeData>\n')
        for n in range(0, N + 1):
            X.append(f'\t\t<SurfaceData name="P{n}" surface="TissueBottom">\n')
            X += [f'\t\t\t<face lid="{i+1}">{v:.7g}</face>\n' for i, v in enumerate(Pface[n])]
            X.append('\t\t</SurfaceData>\n')
        X.append('\t</MeshData>\n')
        # ---- global BCs
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
\t\t<bc name="sides_drained_x" node_set="SideXp" type="zero fluid pressure"/>
\t\t<bc name="sides_drained_y" node_set="SideYp" type="zero fluid pressure"/>
\t</Boundary>
\t<Rigid>
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
""")
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
        # ---- steps (one per coupling window)
        X.append("\t<Step>\n")
        t_prev = 0.0
        lcs = []
        for n in range(1, N + 1):
            t1 = times[n - 1]
            dt = t1 - t_prev
            nst = 6
            up, dn = 2 * n - 1, 2 * n
            lcs.append((up, [(t_prev, 0), (t1, 1)]))
            lcs.append((dn, [(t_prev, 1), (t1, 0)]))
            X.append(f"""\t\t<step id="{n}" name="window{n}">
\t\t\t<Control>
\t\t\t\t<analysis>TRANSIENT</analysis>
\t\t\t\t<time_steps>{nst}</time_steps>
\t\t\t\t<step_size>{dt/nst:.6g}</step_size>
\t\t\t\t<plot_level>PLOT_MUST_POINTS</plot_level>
\t\t\t\t<output_level>OUTPUT_MUST_POINTS</output_level>
\t\t\t\t<time_stepper type="default">
\t\t\t\t\t<max_retries>12</max_retries>
\t\t\t\t\t<opt_iter>12</opt_iter>
\t\t\t\t\t<dtmin>{dt*1e-5:.6g}</dtmin>
\t\t\t\t\t<dtmax>{dt/nst:.6g}</dtmax>
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
\t\t\t</Control>
\t\t\t<Boundary>
\t\t\t\t<bc name="film_p{n}" node_set="Bottom" type="prescribed fluid pressure">
\t\t\t\t\t<value lc="{up}" type="map">dP{n}</value>
\t\t\t\t\t<relative>1</relative>
\t\t\t\t</bc>
\t\t\t</Boundary>
\t\t\t<Loads>
\t\t\t\t<surface_load name="film_load_in{n}" surface="TissueBottom" type="pressure">
\t\t\t\t\t<pressure lc="{up}" type="map">P{n}</pressure>
\t\t\t\t\t<symmetric_stiffness>0</symmetric_stiffness>
\t\t\t\t\t<linear>0</linear>
\t\t\t\t</surface_load>
\t\t\t\t<surface_load name="film_load_out{n}" surface="TissueBottom" type="pressure">
\t\t\t\t\t<pressure lc="{dn}" type="map">P{n-1}</pressure>
\t\t\t\t\t<symmetric_stiffness>0</symmetric_stiffness>
\t\t\t\t\t<linear>0</linear>
\t\t\t\t</surface_load>
\t\t\t</Loads>
\t\t</step>
""")
            t_prev = t1
        X.append("\t</Step>\n\t<LoadData>\n")
        for lc, pts in lcs:
            X.append(f'\t\t<load_controller id="{lc}" name="lc{lc}" type="loadcurve">\n'
                     '\t\t\t<interpolate>LINEAR</interpolate>\n\t\t\t<extend>CONSTANT</extend>\n\t\t\t<points>\n')
            X += [f"\t\t\t\t<pt>{a:.8g},{v}</pt>\n" for a, v in pts]
            X.append("\t\t\t</points>\n\t\t</load_controller>\n")
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
\t\t\t<rigid_body_data file="drain_force.txt" data="Fx;Fy;Fz" delim=",">2</rigid_body_data>
\t\t</logfile>
\t</Output>
</febio_spec>
""")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as f:
            f.write("".join(X))
        json.dump(dict(times=times.tolist(), n_tissue_nodes=int(nT), n_hex=int(nE),
                       n_drain_quads=int(len(self.drain_quads)), n_bottom_faces=int(len(self.bottom))),
                  open(os.path.join(os.path.dirname(os.path.abspath(path)), "febio_info.json"), "w"), indent=1)
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

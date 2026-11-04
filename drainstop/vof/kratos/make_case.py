"""Kratos Multiphysics two-fluid case of the drain at one pool level (mesh, materials, parameters).

    python vof/kratos/make_case.py rec_tilt10 8 [--hole 0.25] [--end 0.08] [--out vof/kratos/runs]

Geometry (gmsh, OpenCASCADE), in the drain frame (x from the tip, z up when lying flat), mm:
- elliptic skin 17.85 x 9.67 mm, 0.6 mm wall, closed tip (0.3 mm cap), end face 0.4 mm with the
  centred Ø4.24 mm bore, and a straight outlet tube (Ø4.24 mm inside, 0.8 mm wall) to 16.8 mm
  beyond the end face, where the suction acts;
- every opening of the model's pattern drilled along the skin normal at its centre (x, theta), with
  the opening's area (half-area edge notches as round openings of equal area);
- a trough 5.5 mm wider than the drain on each side, 5 mm beyond the tip, 4 mm beyond the end face
  (a wall the tube passes through), floor 2.5 mm below the bottom line, open top.
The 10 deg tilt enters through gravity only, as in the OpenFOAM cases in vof/cases.

Tetrahedral mesh: --hole mm at the opening rims, the bore and the caps, 0.5 mm on the skin,
0.8 mm inside the drain, 1.6 mm in the pool. Writes <run>/drain.mdpa (m), materials.json,
ProjectParameters.json and model_state.json (the model's prediction at that level).

Start: liquid (level set < 0) below the pool level h outside the drain and below the model's inner
level zeta(h) inside it, air above.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOF = os.path.dirname(HERE)
ROOT = os.path.dirname(VOF)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_study  # noqa: E402

TILT = 10.0
SPECS = {"usz_tilt10": dict(layout="usz", d=1.0, family="reference", tilt=TILT), "rec_tilt10": None}
T_TIP, T_END = 0.3, 0.4            # mm, tip cap and end face (as in vof/make_cases.py)
TUBE_WALL, TUBE_LEN = 0.8, 16.8    # mm, outlet tube wall and length beyond the end face
GAP_SIDE, GAP_TIP, GAP_END, GAP_FLOOR = 5.5, 5.0, 4.0, 2.5
SIZE_SKIN, SIZE_LUMEN, SIZE_POOL = 0.5, 0.8, 1.6


def model_and_state(case, h):
    spec = SPECS[case] or dict(run_study.best_spec(), tilt=TILT)
    m = run_study.make_model(spec)
    br = np.zeros(m.pattern.n, bool)
    for hh in np.arange(m.h_start(), h, -0.25):              # march down so openings breach in order
        br = m.state(hh + m.Z_min, br)["breached"]
    st = m.state(h + m.Z_min, br)
    state = dict(case=case, tilt=TILT, h=h, mode=st["mode"], zeta=float(st["zeta"] - m.Z_min),
                 pg=float(st["pg"]), qL_ml_min=st["qL"] * 6e7, qG_l_min=st["qG"] * 6e4,
                 n_open=int(st["breached"].sum()), z_lip=float(m.Z_lip - m.Z_min),
                 h_stop=float(m.descend(dh=0.1)["h_stop"]))
    return m, state


END_CLEARANCE = 0.35   # mm of wall kept between an opening and the tip cap / end face


def hole_frames(m):
    """Centre on the outer skin, outward unit normal (y, z) and radius of every opening (mm).

    Openings closer than END_CLEARANCE to the tip cap or the end face are moved inward until they
    are not (USZ: the first column by 0.25 mm, the last by 0.15 mm), so that no wall sliver thinner
    than a tetrahedron is left there."""
    pat, a, b = m.pattern, m.drain.a, m.drain.b
    th = pat.theta
    ny, nz = np.sin(th) / a, -np.cos(th) / b
    nn = np.hypot(ny, nz)
    r = 0.5 * pat.d * np.sqrt(pat.area_scale)
    x = np.clip(pat.x, T_TIP + r + END_CLEARANCE, m.drain.length - T_END - r - END_CLEARANCE)
    c = np.column_stack([x, a * np.sin(th), b * (1 - np.cos(th))])
    n = np.column_stack([np.zeros_like(th), ny / nn, nz / nn])
    return c, n, r


def build_geometry(gmsh, m, z_top):
    occ = gmsh.model.occ
    dr = m.drain
    a, b, t, L, rp = dr.a, dr.b, dr.wall, dr.length, dr.port_d / 2

    def ell_cyl(x0, x1, ra, rb):
        cv = occ.addEllipse(x0, 0, b, ra, rb, zAxis=[1, 0, 0], xAxis=[0, 1, 0])
        s = occ.addPlaneSurface([occ.addCurveLoop([cv])])
        return occ.extrude([(2, s)], x1 - x0, 0, 0)[1]
    outer = ell_cyl(0.0, L, a, b)
    lumen = ell_cyl(T_TIP, L - T_END, a - t, b - t)
    c, n, r = hole_frames(m)
    holes = []
    for ci, ni, ri in zip(c, n, r):
        p0 = ci + 0.5 * ni
        d = -(t + 1.2) * ni
        holes.append((3, occ.addCylinder(*p0, *d, ri)))
    bore = (3, occ.addCylinder(L - T_END - 0.2, 0, b, T_END + 0.4, 0, 0, rp))
    drain, _ = occ.cut([outer], [lumen] + holes + [bore])
    box = (3, occ.addBox(-GAP_TIP, -(a + GAP_SIDE), -GAP_FLOOR, L + GAP_TIP + GAP_END, 2 * (a + GAP_SIDE),
                         z_top + GAP_FLOOR))
    tube_in = (3, occ.addCylinder(L - 0.1, 0, b, TUBE_LEN + 0.1, 0, 0, rp))
    tube_out = (3, occ.addCylinder(L, 0, b, TUBE_LEN, 0, 0, rp + TUBE_WALL))
    tube_wall, _ = occ.cut([tube_out], [(3, occ.addCylinder(L - 0.2, 0, b, TUBE_LEN + 0.4, 0, 0, rp))])
    fluid0, _ = occ.fuse([box], [tube_in])
    fluid, _ = occ.cut(fluid0, drain + tube_wall)
    occ.synchronize()
    return fluid, dict(L=L, b=b, rp=rp, z_top=z_top, x_out=L + TUBE_LEN)


def size_callback_factory(m, h_hole, size_skin=SIZE_SKIN):
    """Target edge length (mm): fine at opening rims, bore and caps, coarser on the skin, inside
    the drain and in the pool. Holes are looked up in an x-window for speed."""
    dr = m.drain
    a, b, t, L, rp = dr.a, dr.b, dr.wall, dr.length, dr.port_d / 2
    c, n, r = hole_frames(m)
    mid = c - 0.5 * t * n
    order = np.argsort(mid[:, 0])
    mid, rr = mid[order], r[order]
    xs = mid[:, 0]

    def size(dim, tag, x, y, z, lc):
        # opening rims
        i0, i1 = np.searchsorted(xs, x - 3.0), np.searchsorted(xs, x + 3.0)
        s = SIZE_POOL
        if i1 > i0:
            d = np.sqrt((mid[i0:i1, 0] - x) ** 2 + (mid[i0:i1, 1] - y) ** 2 + (mid[i0:i1, 2] - z) ** 2)
            dd = float(np.min(d - rr[i0:i1]))
            s = min(s, h_hole + max(dd - 0.35, 0.0) * 0.45)
        # bore, end face and tube entrance
        dbore = math.hypot(math.hypot(y, z - b) - rp, max(abs(x - (L - T_END / 2)) - T_END, 0.0))
        s = min(s, h_hole + max(dbore - 0.3, 0.0) * 0.4)
        if x > L and math.hypot(y, z - b) < rp + 0.1:
            s = min(s, 0.45)
        # tip cap
        if x < 1.5 and math.hypot(y / a, (z - b) / b) < 1.1:
            s = min(s, max(h_hole, 0.3))
        # skin and interior
        q = math.hypot(y / a, (z - b) / b)
        dskin = abs(q - 1.0) * b
        s = min(s, size_skin + max(dskin - 0.6, 0.0) * 0.5)
        if q < 1.0 and -0.5 < x < L + 0.5:
            s = min(s, SIZE_LUMEN)
        return min(s, lc)                        # lc: gmsh's curvature size (rims on the curved flanks)
    return size


def classify_surfaces(gmsh, info):
    """Physical groups: Outlet (tube end), Atmosphere (trough top), Walls (the rest)."""
    out, atm, walls = [], [], []
    for _, tag in gmsh.model.getEntities(2):
        x0, y0, z0, x1, y1, z1 = gmsh.model.getBoundingBox(2, tag)
        if abs(x0 - info["x_out"]) < 1e-3 and abs(x1 - info["x_out"]) < 1e-3:
            out.append(tag)
        elif abs(z0 - info["z_top"]) < 1e-3 and abs(z1 - info["z_top"]) < 1e-3:
            atm.append(tag)
        else:
            walls.append(tag)
    return dict(Outlet=out, Atmosphere=atm, Walls=walls)


def mesh(case, m, h_hole, z_top, msh_path, size_skin=SIZE_SKIN, curv=24):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.NumThreads", os.cpu_count() or 4)
    gmsh.model.add(case)
    t0 = time.time()
    fluid, info = build_geometry(gmsh, m, z_top)
    print(f"  geometry: {len(fluid)} volume(s), {len(gmsh.model.getEntities(2))} surfaces, {time.time() - t0:.0f} s")
    groups = classify_surfaces(gmsh, info)
    for name, tags in groups.items():
        gmsh.model.addPhysicalGroup(2, tags, name=name)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in fluid], name="Fluid")
    gmsh.model.mesh.setSizeCallback(size_callback_factory(m, h_hole, size_skin))
    # curvature sizing: about 24 edges around an opening rim, so that rims on the strongly curved
    # flanks of the ellipse (USZ openings at +-56 and +-67 deg) are not cut by the skin triangles
    for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                 ("Mesh.MeshSizeFromCurvature", curv), ("Mesh.Algorithm", 6), ("Mesh.Algorithm3D", 10),
                 ("Mesh.Optimize", 1), ("Mesh.MeshSizeMin", 0.1), ("Mesh.MeshSizeMax", SIZE_POOL)):
        gmsh.option.setNumber(k, v)
    t0 = time.time()
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:                       # noqa: BLE001  HXT can fail where Delaunay works
        print(f"  HXT failed ({e}); retrying the volume mesh with Delaunay")
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
    print(f"  mesh: {time.time() - t0:.0f} s")
    nodes = gmsh.model.mesh.getNodes()
    if nodes[0].size == 0 or not any(t_ == 4 for t_ in gmsh.model.mesh.getElements(3)[0]):
        gmsh.finalize()
        raise RuntimeError("volume mesh failed (see the gmsh messages above)")
    gmsh.write(msh_path)
    tags, xyz = nodes[0], nodes[1].reshape(-1, 3)
    idx = np.full(int(tags.max()) + 1, -1, np.int64)
    idx[tags.astype(np.int64)] = np.arange(tags.size)
    et, _, en = gmsh.model.mesh.getElements(3)
    tets = idx[np.concatenate([e for t_, e in zip(et, en) if t_ == 4]).astype(np.int64)].reshape(-1, 4)
    surf = {}
    for name in groups:
        tris = []
        for tag in groups[name]:
            et, _, en = gmsh.model.mesh.getElements(2, tag)
            for t_, e in zip(et, en):
                if t_ == 2:
                    tris.append(idx[e.astype(np.int64)].reshape(-1, 3))
        surf[name] = np.concatenate(tris) if tris else np.zeros((0, 3), np.int64)
    gmsh.finalize()
    return xyz, tets, surf


def orient_outward(xyz, tets, tris):
    """Flip boundary triangles so that their normals point out of the fluid."""
    if len(tris) == 0:
        return tris
    N = np.int64(xyz.shape[0])
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    opp = np.concatenate([tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]])
    fs = np.sort(faces, axis=1).astype(np.int64)
    key = (fs[:, 0] * N + fs[:, 1]) * N + fs[:, 2]
    o = np.argsort(key)
    key, opp = key[o], opp[o]
    ts = np.sort(tris, axis=1).astype(np.int64)
    tkey = (ts[:, 0] * N + ts[:, 1]) * N + ts[:, 2]
    pos = np.searchsorted(key, tkey)
    pos = np.clip(pos, 0, key.size - 1)
    ok = key[pos] == tkey
    p = xyz[tris]
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", nrm, xyz[opp[pos]] - p[:, 0]) > 0
    out = tris.copy()
    flip = ok & inward
    out[flip] = out[flip][:, [0, 2, 1]]
    if (~ok).any():
        print(f"  warning: {int((~ok).sum())} boundary triangles without a tetrahedron")
    return out


def initial_distance(m, xyz_mm, h, zeta):
    """Signed level set (m): negative in the liquid. Pool below h outside the drain, below zeta
    inside the drain interior, the bore and the tube."""
    dr = m.drain
    a, b, t, L, rp = dr.a, dr.b, dr.wall, dr.length, dr.port_d / 2
    x, y, z = xyz_mm.T
    al = math.radians(TILT)
    Zw = x * math.sin(al) + z * math.cos(al)                  # world height above the lowest skin point
    inside = ((x > T_TIP - 1e-6) & (x < L - T_END + 1e-6) & (np.hypot(y / (a - t), (z - b) / (b - t)) < 1 + 1e-6)) | \
             ((x >= L - T_END - 1e-6) & (np.hypot(y, z - b) < rp + 1e-6))
    lev = np.where(inside, zeta, h)
    return (Zw - lev) * 1e-3


def write_mdpa(path, xyz_mm, tets, surf, dist):
    xyz = xyz_mm * 1e-3
    n = xyz.shape[0]
    with open(path, "w", newline="\n") as f:
        f.write("Begin ModelPartData\nEnd ModelPartData\n\n")
        for p in (0, 1, 2):
            f.write(f"Begin Properties {p}\nEnd Properties\n")
        f.write("\nBegin Nodes\n")
        np.savetxt(f, np.column_stack([np.arange(1, n + 1), xyz]), fmt="%d %.9e %.9e %.9e")
        f.write("End Nodes\n\nBegin Elements Element3D4N\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(tets) + 1), np.ones(len(tets), int), tets + 1]),
                   fmt="%d")
        f.write("End Elements\n\nBegin Conditions SurfaceCondition3D3N\n")
        cid, ranges = 1, {}
        for name, tris in surf.items():
            if len(tris):
                ids = np.arange(cid, cid + len(tris))
                np.savetxt(f, np.column_stack([ids, np.zeros(len(tris), int), tris + 1]), fmt="%d")
                ranges[name] = ids
                cid += len(tris)
        f.write("End Conditions\n\nBegin NodalData DISTANCE\n")
        np.savetxt(f, np.column_stack([np.arange(1, n + 1), np.zeros(n, int), dist]), fmt="%d %d %.9e")
        f.write("End NodalData\n\n")
        f.write("Begin SubModelPart Fluid\n Begin SubModelPartNodes\n")
        np.savetxt(f, np.arange(1, n + 1), fmt="%d")
        f.write(" End SubModelPartNodes\n Begin SubModelPartElements\n")
        np.savetxt(f, np.arange(1, len(tets) + 1), fmt="%d")
        f.write(" End SubModelPartElements\n Begin SubModelPartConditions\n End SubModelPartConditions\nEnd SubModelPart\n")
        for name, tris in surf.items():
            f.write(f"Begin SubModelPart {name}\n Begin SubModelPartNodes\n")
            np.savetxt(f, np.unique(tris) + 1, fmt="%d")
            f.write(" End SubModelPartNodes\n Begin SubModelPartElements\n End SubModelPartElements\n")
            f.write(" Begin SubModelPartConditions\n")
            if name in ranges:
                np.savetxt(f, ranges[name], fmt="%d")
            f.write(" End SubModelPartConditions\nEnd SubModelPart\n")


def write_parameters(run, end_time, out_dt, m, cfl, dt_max, dt_fixed=None):
    al = math.radians(TILT)
    g_dir = [-math.sin(al), 0.0, -math.cos(al)]
    law = {"name": "Newtonian3DLaw"}
    materials = {"properties": [
        {"model_part_name": "FluidModelPart.Fluid", "properties_id": 1,
         "Material": {"constitutive_law": law,
                      "Variables": {"DENSITY": m.fluid.rho, "DYNAMIC_VISCOSITY": m.fluid.mu}, "Tables": {}}},
        {"model_part_name": "FluidModelPart.Fluid", "properties_id": 2,
         "Material": {"constitutive_law": law,
                      "Variables": {"DENSITY": m.gas.rho, "DYNAMIC_VISCOSITY": m.gas.mu}, "Tables": {}}}]}
    json.dump(materials, open(os.path.join(run, "materials.json"), "w"), indent=2)
    ps = m.suction.p_set
    params = {
        "problem_data": {"problem_name": "drain", "parallel_type": "OpenMP", "echo_level": 0,
                         "start_time": 0.0, "end_time": end_time},
        "output_processes": {"vtk_output": [{
            "python_module": "vtk_output_process", "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": {"model_part_name": "FluidModelPart.Fluid", "output_control_type": "time",
                           "output_interval": out_dt, "file_format": "binary", "output_precision": 7,
                           "output_sub_model_parts": False, "output_path": "vtk_output",
                           "save_output_files_in_folder": True,
                           "nodal_solution_step_data_variables": ["VELOCITY", "PRESSURE", "DISTANCE"]}}]},
        "solver_settings": {
            "model_part_name": "FluidModelPart", "domain_size": 3, "solver_type": "two_fluids",
            "model_import_settings": {"input_type": "mdpa", "input_filename": "drain"},
            "material_import_settings": {"materials_filename": "materials.json"},
            "echo_level": 1, "compute_reactions": False, "maximum_iterations": 3,
            "relative_velocity_tolerance": 1e-3, "absolute_velocity_tolerance": 1e-5,
            "relative_pressure_tolerance": 1e-3, "absolute_pressure_tolerance": 1e-5,
            "volume_model_part_name": "Fluid", "skin_parts": ["Walls", "Outlet", "Atmosphere"],
            "no_skin_parts": [],
            # fixed step (default): the implicit BDF2 solver is stable well above CFL 1 in the air jets,
            # and the quasi-steady flows are what is wanted; the liquid surface moves < 1 cell per step
            "time_stepping": ({"automatic_time_step": False, "time_step": dt_fixed} if dt_fixed else
                              {"automatic_time_step": True, "CFL_number": cfl, "minimum_delta_time": 1e-7,
                               "maximum_delta_time": dt_max}),
            "formulation": {"dynamic_tau": 1.0, "surface_tension": False, "mass_source": True},
            # at most 4 sub-steps: unlimited sub-stepping follows the air jets (CFL ~10) far from
            # the interface and more than doubles the cost; near the interface the flow is slow
            "levelset_convection_settings": {"max_CFL": 1.0, "max_substeps": 4,
                                             "eulerian_error_compensation": False,
                                             "element_type": "levelset_convection_supg"},
            "distance_reinitialization": "parallel", "parallel_redistance_max_layers": 15,
            "distance_modification_settings": {"model_part_name": "FluidModelPart.Fluid",
                                               "distance_threshold": 1e-3,
                                               "check_at_each_time_step": True,
                                               "avoid_almost_empty_elements": True,
                                               "deactivate_full_negative_elements": False},
            "linear_solver_settings": {"solver_type": "amgcl", "max_iteration": 200, "tolerance": 1e-6,
                                       "smoother_type": "ilu0", "krylov_type": "lgmres",
                                       "coarsening_type": "aggregation", "scaling": False,
                                       "use_block_matrices_if_possible": True, "verbosity": 0},
            # Kratos' acceleration limitation caps every nodal acceleration at 5 g; the air jets
            # through the openings and the bore accelerate at ~1e5 m/s2, so it must be off here
            "acceleration_limitation": False},
        "processes": {
            "initial_conditions_process_list": [],
            "boundary_conditions_process_list": [
                {"python_module": "apply_noslip_process", "kratos_module": "KratosMultiphysics.FluidDynamicsApplication",
                 "process_name": "ApplyNoSlipProcess", "Parameters": {"model_part_name": "FluidModelPart.Walls"}},
                {"python_module": "apply_outlet_process", "kratos_module": "KratosMultiphysics.FluidDynamicsApplication",
                 "process_name": "ApplyOutletProcess",
                 "Parameters": {"model_part_name": "FluidModelPart.Outlet", "variable_name": "PRESSURE",
                                "constrained": True, "value": ps, "hydrostatic_outlet": False, "h_top": 0.0,
                                "outlet_inflow_contribution": True}},
                {"python_module": "apply_outlet_process", "kratos_module": "KratosMultiphysics.FluidDynamicsApplication",
                 "process_name": "ApplyOutletProcess",
                 "Parameters": {"model_part_name": "FluidModelPart.Atmosphere", "variable_name": "PRESSURE",
                                "constrained": True, "value": 0.0, "hydrostatic_outlet": False, "h_top": 0.0,
                                "outlet_inflow_contribution": True}}],
            "gravity": [{"python_module": "assign_vector_by_direction_process", "kratos_module": "KratosMultiphysics",
                         "process_name": "AssignVectorByDirectionProcess",
                         "Parameters": {"model_part_name": "FluidModelPart.Fluid", "variable_name": "BODY_FORCE",
                                        "modulus": 9.81, "constrained": False, "direction": g_dir}}],
            "auxiliar_process_list": []}}
    json.dump(params, open(os.path.join(run, "ProjectParameters.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(SPECS))
    ap.add_argument("level", type=float)
    ap.add_argument("--hole", type=float, default=0.25, help="mm, cell size at the opening rims and the bore")
    ap.add_argument("--skin", type=float, default=SIZE_SKIN, help="mm, cell size on the skin")
    ap.add_argument("--curv", type=int, default=24, help="edges per 2*pi of curvature (opening rims)")
    ap.add_argument("--end", type=float, default=0.15, help="s of simulated time")
    ap.add_argument("--out-dt", type=float, default=0.005, help="s between field outputs")
    ap.add_argument("--cfl", type=float, default=4.0, help="CFL number of the implicit time step")
    ap.add_argument("--dt-max", type=float, default=5e-5)
    ap.add_argument("--dt", type=float, default=5e-5, help="fixed time step (s); 0 = automatic from --cfl")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"{a.case}_h{a.level:g}"
    run = os.path.join(a.out, name)
    os.makedirs(run, exist_ok=True)
    m, st = model_and_state(a.case, a.level)
    json.dump(st, open(os.path.join(run, "model_state.json"), "w"), indent=1)
    print(f"{name}: model zeta {st['zeta']:.2f} mm, pg {st['pg']:.1f} Pa, liquid {st['qL_ml_min']:.0f} mL/min, "
          f"air {st['qG_l_min']:.1f} L/min")
    ca = math.cos(math.radians(TILT))
    z_top = max(2 * m.drain.b + 3.5, (a.level + 4.0) / ca + 1.0)
    xyz, tets, surf = mesh(a.case, m, a.hole, z_top, os.path.join(run, "drain.msh"), a.skin, a.curv)
    print(f"  {xyz.shape[0]:,} nodes, {tets.shape[0]:,} tetrahedra; "
          + ", ".join(f"{k} {len(v):,}" for k, v in surf.items()))
    surf = {k: orient_outward(xyz, tets, v) for k, v in surf.items()}
    dist = initial_distance(m, xyz, a.level, st["zeta"])
    write_mdpa(os.path.join(run, "drain.mdpa"), xyz, tets, surf, dist)
    write_parameters(run, a.end, a.out_dt, m, a.cfl, a.dt_max, a.dt or None)
    json.dump(dict(case=a.case, level=a.level, hole=a.hole, skin=a.skin, curv=a.curv, z_top=z_top, nodes=int(xyz.shape[0]),
                   tets=int(tets.shape[0])), open(os.path.join(run, "mesh_info.json"), "w"), indent=1)
    print("wrote", run)


if __name__ == "__main__":
    main()

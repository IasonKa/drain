"""
Flow model of the USZ sheath without the internal tube, with an irrigation outlet in the tip.

Fluid domain (local frame, mm; Y along the drain, tip at +Y, suction end at -Y):
  lumen     the real rib profile of the sheath interior (sliced from the USZ CAD), Y = -59.5 .. 59.9
  nozzle    irrigation outlet, d = 1.0 mm, on the axis in the tip cap, blowing towards the suction end
  outflow   round outflow tube, inner d = 4.2 mm, 12 mm long; suction pressure at its end
  holes     d = 1.0 mm openings through the top skin (only the open ones; blocked holes are wall)

  <kratos python> make_flow_case.py <case> [--q 5] [--h 0.55]
     case: blocked | tip | mid | conn       q: irrigation flow, mL/min
Writes runs/<case>/{flow.mdpa, ProjectParameters.json, materials.json, mesh.npz, case.json}
"""
import argparse, json, os, sys
import numpy as np
import gmsh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from make_design import load_local, XC, ZC

YA, YB = -59.5, 59.9            # lumen: outflow-tube mouth .. inner face of the tip cap
LT, R_TUBE = 12.0, 2.1          # outflow tube
Y_FN, R_FN = -67.14, 2.8        # narrow end of the V (funnel) and its radius there
FUN_A, FUN_B = 8.3, 4.2         # funnel half-axes at the sheath end
LN, R_NOZ = 2.0, 0.5            # irrigation nozzle
R_HOLE = 0.5
Z_BATH, H_BATH = 0.1, 4.0       # bath pocket over the open holes: bottom just above the skin, height
A_OUT, B_OUT = 8.925, 4.835     # skin ellipse
P_SUCTION = -933.0              # Pa (-7 mmHg), as in the drain-stop study
RHO, MU = 1000.0, 1.0e-3        # saline
HOLES = {"blocked": [],
         "tip": [(0.0, 54.0), (3.3, 54.0), (0.0, 49.5), (3.3, 49.5)],
         "mid": [(0.0, 0.0), (3.3, 0.0), (0.0, 4.5), (3.3, 4.5)],
         "conn": [(0.0, -54.0), (3.3, -54.0), (0.0, -49.5), (3.3, -49.5)]}


def z_skin(x):
    return ZC + B_OUT * np.sqrt(1 - ((x - XC) / A_OUT) ** 2)


def _area(P):
    return 0.5 * abs(np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1]))


def lumen_profile():
    sl = load_local().slice(normal=(0, 1, 0), origin=(0, 1.1, 0)).strip(join=True)
    loops, lines, i = [], sl.lines, 0
    while i < len(lines):
        k = lines[i]
        P = sl.points[lines[i + 1:i + 1 + k]][:, [0, 2]]
        i += k + 1
        if np.hypot(P[:, 0] - XC, P[:, 1] - ZC).max() > 5:
            loops.append(P)
    P = sorted(loops, key=_area)[-2]                      # largest = skin, second largest = lumen outline
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    keep = [P[0]]
    for q in P[1:]:
        if np.hypot(*(q - keep[-1])) > 0.03:
            keep.append(q)
    P = np.array(keep)
    out = []
    n = len(P)
    for j in range(n):                                    # drop collinear points
        a, b, c = P[j - 1], P[j], P[(j + 1) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) > 0.01 * np.hypot(*(b - a)) * np.hypot(*(c - b)) + 1e-9:
            out.append(b)
    P = np.array(out)
    return P, _area(np.vstack([P, P[:1]]))


def orient(xyz, tets, tris):
    a, b, c, d = (xyz[tets[:, k]] for k in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    N = np.int64(len(xyz) + 1)
    fid = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    F = tets[:, fid].reshape(-1, 3)
    opp = tets.ravel()

    def key(T):
        S = np.sort(T, axis=1).astype(np.int64)
        return S[:, 0] * N * N + S[:, 1] * N + S[:, 2]
    kf = key(F)
    o = np.argsort(kf)
    kf = kf[o]
    j = np.searchsorted(kf, key(tris))
    assert (kf[j] == key(tris)).all()
    po = xyz[opp[o[j]]]
    p0, p1, p2 = (xyz[tris[:, k]] for k in range(3))
    inward = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), po - p0) > 0
    tris[inward] = tris[inward][:, [0, 2, 1]]
    return tets, tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(HOLES))
    ap.add_argument("--q", type=float, default=250.0 / 60)   # 250 mL/h
    ap.add_argument("--h", type=float, default=0.55)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    run = os.path.join(HERE, "runs", a.name or a.case)
    os.makedirs(run, exist_ok=True)
    P, area = lumen_profile()
    holes = [(x, y, float(z_skin(x))) for x, y in HOLES[a.case]]
    print("lumen outline points", len(P), "area %.1f mm2" % area, flush=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    pts = [occ.addPoint(x, YA, z) for x, z in P]
    lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    s = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    vol = [d for d in occ.extrude([(2, s)], 0, YB - YA, 0) if d[0] == 3]
    # V connection: funnel from the lumen (ellipse) down to the suction tube, then the tube bore
    e1 = occ.addCurveLoop([occ.addEllipse(XC, YA + 0.3, ZC, FUN_A, FUN_B, zAxis=[0, 1, 0], xAxis=[1, 0, 0])])
    e2 = occ.addCurveLoop([occ.addCircle(XC, Y_FN, ZC, R_FN, zAxis=[0, 1, 0], xAxis=[1, 0, 0])])
    funnel = [d for d in occ.addThruSections([e1, e2], makeSolid=True, makeRuled=True) if d[0] == 3]
    tools = funnel + [(3, occ.addCylinder(XC, Y_FN + 0.3, ZC, 0, -(LT + 0.3), 0, R_TUBE)),
                      (3, occ.addCylinder(XC, YB - 0.5, ZC, 0, LN + 0.5, 0, R_NOZ))]
    for x, y, zs in holes:
        tools.append((3, occ.addCylinder(x, y, zs - 2.4, 0, 0, Z_BATH + 0.2 - (zs - 2.4), R_HOLE)))
    if holes:
        # a pocket of the surrounding fluid over the open holes; its sides and top are the open boundary (p = 0).
        # A pressure boundary right at the hole mouths is unstable when fluid is drawn in through it.
        hx, hy = [h[0] for h in holes], [h[1] for h in holes]
        tools.append((3, occ.addBox(min(hx) - 3, min(hy) - 3, Z_BATH, max(hx) - min(hx) + 6, max(hy) - min(hy) + 6, H_BATH)))
    vol, _ = occ.fuse(vol, tools)
    occ.synchronize()
    assert len(vol) == 1, vol

    def size(dim, tag, x, y, z, lc):
        h = a.h
        r = np.hypot(x - XC, z - ZC)
        if y > YB - 0.2:
            h = 0.14
        else:
            d = YB - y                                    # jet region behind the nozzle
            if d < 40 and r < 1.0 + 0.06 * d:
                h = min(h, 0.16 + 0.008 * d)
        if y < YA + 1:
            h = min(h, 0.4)
        for hx, hy, hz in holes:
            dd = np.sqrt((x - hx) ** 2 + (y - hy) ** 2 + max(0.0, (hz - 2.0) - z) ** 2)
            h = min(h, 0.16 + 0.12 * max(0.0, dd - 0.6))
        return h
    gmsh.model.mesh.setSizeCallback(size)
    for k in ("MeshSizeFromPoints", "MeshSizeFromCurvature", "MeshSizeExtendFromBoundary"):
        gmsh.option.setNumber("Mesh." + k, 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)
    gmsh.option.setNumber("General.NumThreads", 8)
    gmsh.model.mesh.generate(3)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    idx = np.zeros(int(tags.max()) + 1, int)
    idx[tags.astype(int)] = np.arange(len(tags))
    tets = idx[gmsh.model.mesh.getElementsByType(4)[1].astype(int)].reshape(-1, 4)
    tris = idx[gmsh.model.mesh.getElementsByType(2)[1].astype(int)].reshape(-1, 3)
    gmsh.finalize()
    used = np.unique(tets)
    remap = -np.ones(len(xyz), int)
    remap[used] = np.arange(len(used))
    xyz, tets, tris = xyz[used], remap[tets], remap[tris]
    tets, tris = orient(xyz, tets, tris)
    c = xyz[tris].mean(axis=1)
    lab = np.zeros(len(tris), int)                        # 0 wall, 1 inlet, 2 outlet, 3+k hole k
    lab[c[:, 1] > YB + LN - 1e-4] = 1
    lab[c[:, 1] < Y_FN - LT + 1e-4] = 2
    lab[c[:, 2] > Z_BATH + 1e-4] = 3                      # sides and top of the bath pocket
    surf = {"Inlet": tris[lab == 1], "Outlet": tris[lab == 2], "Walls": tris[lab == 0]}
    if holes:
        surf["Holes"] = tris[lab >= 3]
    print("nodes", len(xyz), "tets", len(tets), {k: len(v) for k, v in surf.items()}, flush=True)
    np.savez(os.path.join(run, "mesh.npz"), xyz=xyz, tets=tets, tris=tris, lab=lab, profile=P)

    X = xyz * 1e-3
    n = len(X)
    with open(os.path.join(run, "flow.mdpa"), "w", newline="\n") as f:
        f.write("Begin ModelPartData\nEnd ModelPartData\n\nBegin Properties 0\nEnd Properties\n"
                "Begin Properties 1\nEnd Properties\n\nBegin Nodes\n")
        np.savetxt(f, np.column_stack([np.arange(1, n + 1), X]), fmt="%d %.9e %.9e %.9e")
        f.write("End Nodes\n\nBegin Elements Element3D4N\n")
        np.savetxt(f, np.column_stack([np.arange(1, len(tets) + 1), np.ones(len(tets), int), tets + 1]), fmt="%d")
        f.write("End Elements\n\nBegin Conditions SurfaceCondition3D3N\n")
        cid, ranges = 1, {}
        for name, t in surf.items():
            ids = np.arange(cid, cid + len(t))
            ranges[name] = ids
            cid += len(t)
            np.savetxt(f, np.column_stack([ids, np.zeros(len(t), int), t + 1]), fmt="%d")
        f.write("End Conditions\n\nBegin SubModelPart Fluid\n Begin SubModelPartNodes\n")
        np.savetxt(f, np.arange(1, n + 1), fmt="%d")
        f.write(" End SubModelPartNodes\n Begin SubModelPartElements\n")
        np.savetxt(f, np.arange(1, len(tets) + 1), fmt="%d")
        f.write(" End SubModelPartElements\n Begin SubModelPartConditions\n End SubModelPartConditions\nEnd SubModelPart\n")
        for name, t in surf.items():
            f.write(f"Begin SubModelPart {name}\n Begin SubModelPartNodes\n")
            np.savetxt(f, np.unique(t) + 1, fmt="%d")
            f.write(" End SubModelPartNodes\n Begin SubModelPartElements\n End SubModelPartElements\n"
                    " Begin SubModelPartConditions\n")
            np.savetxt(f, ranges[name], fmt="%d")
            f.write(" End SubModelPartConditions\nEnd SubModelPart\n")

    # inlet speed: rim nodes are held at zero by the wall, so scale the core speed to deliver exactly q
    ti = surf["Inlet"]
    rim = np.isin(ti, np.unique(surf["Walls"]))
    ai = 0.5 * np.linalg.norm(np.cross(xyz[ti[:, 1]] - xyz[ti[:, 0]], xyz[ti[:, 2]] - xyz[ti[:, 0]]), axis=1)
    eff_area = float((ai * (~rim).mean(axis=1)).sum())            # mm2
    v_in = a.q * 1e-6 / 60 / (eff_area * 1e-6)
    json.dump({"properties": [{"model_part_name": "FluidModelPart.Fluid", "properties_id": 1,
               "Material": {"constitutive_law": {"name": "Newtonian3DLaw"},
                            "Variables": {"DENSITY": RHO, "DYNAMIC_VISCOSITY": a.mu}, "Tables": {}}}]},
              open(os.path.join(run, "materials.json"), "w"), indent=1)
    fd = "KratosMultiphysics.FluidDynamicsApplication"

    def outlet(part, p):
        return {"python_module": "apply_outlet_process", "kratos_module": fd, "process_name": "ApplyOutletProcess",
                "Parameters": {"model_part_name": "FluidModelPart." + part, "variable_name": "PRESSURE",
                               "constrained": True, "value": p, "hydrostatic_outlet": False, "h_top": 0.0,
                               "outlet_inflow_contribution": True}}
    bcs = [{"python_module": "assign_vector_variable_process", "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorVariableProcess",
            "Parameters": {"model_part_name": "FluidModelPart.Inlet", "variable_name": "VELOCITY",
                           "value": [0.0, -v_in, 0.0], "constrained": [True, True, True]}},
           {"python_module": "apply_noslip_process", "kratos_module": fd, "process_name": "ApplyNoSlipProcess",
            "Parameters": {"model_part_name": "FluidModelPart.Walls"}},
           # with open holes the suction is ramped up over 0.2 s: switching it on at once blows the first step up
           outlet("Outlet", f"{P_SUCTION}*(1.0-exp(-t/0.05))" if holes else P_SUCTION)] + ([outlet("Holes", 0.0)] if holes else [])
    params = {"problem_data": {"problem_name": "flow", "parallel_type": "OpenMP", "echo_level": 0,
                               "start_time": 0.0, "end_time": 1.0},
              "output_processes": {},
              "solver_settings": {"model_part_name": "FluidModelPart", "domain_size": 3, "solver_type": "Monolithic",
                                  "model_import_settings": {"input_type": "mdpa", "input_filename": "flow"},
                                  "material_import_settings": {"materials_filename": "materials.json"},
                                  "echo_level": 0, "compute_reactions": False, "maximum_iterations": 4,
                                  "relative_velocity_tolerance": 1e-4, "absolute_velocity_tolerance": 1e-7,
                                  "relative_pressure_tolerance": 1e-4, "absolute_pressure_tolerance": 1e-7,
                                  "volume_model_part_name": "Fluid", "skin_parts": list(surf), "no_skin_parts": [],
                                  "time_scheme": "bdf2",
                                  "time_stepping": {"automatic_time_step": False, "time_step": 0.01},
                                  "formulation": {"element_type": "qsvms", "use_orthogonal_subscales": False,
                                                  "dynamic_tau": 1.0},
                                  "linear_solver_settings": {"solver_type": "amgcl", "max_iteration": 200,
                                                             "tolerance": 1e-6, "smoother_type": "ilu0",
                                                             "krylov_type": "lgmres", "coarsening_type": "aggregation",
                                                             "scaling": False, "use_block_matrices_if_possible": True,
                                                             "verbosity": 0}},
              "processes": {"initial_conditions_process_list": [], "boundary_conditions_process_list": bcs,
                            "gravity": [], "auxiliar_process_list": []}}
    json.dump(params, open(os.path.join(run, "ProjectParameters.json"), "w"), indent=1)
    json.dump(dict(case=a.case, q_ml_min=a.q, v_in=float(v_in), holes=holes, mu=a.mu, rho=RHO, p_suction=P_SUCTION,
                   lumen_area_mm2=float(area), nodes=int(n), tets=int(len(tets))),
              open(os.path.join(run, "case.json"), "w"), indent=1)
    print("written", run)


if __name__ == "__main__":
    main()

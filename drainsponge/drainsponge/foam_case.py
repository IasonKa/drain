"""
OpenFOAM cases for the single-hole window.

kind = "porous"  : ONE-DOMAIN Darcy-Brinkman model, runnable on its own.
                   Free liquid in channel/hole/film + the tissue as a porous cellZone
                   (Darcy drag d = 1/K, K = k*mu). Shows how much liquid comes from the
                   free film and how much is pulled out of the tissue itself.
                   Tissue shape is frozen (initial, or a deformed shape from FEBio /
                   the preview model). Needs the ESI edition (simpleFoam + fvOptions).
kind = "fluid"   : channel/hole/film only. The tissue is a moving boundary
                   (tissue_iface) whose velocity = tissue surface velocity + seepage
                   from FEBio. Used by couple.py; runs on both editions.

Steady, incompressible, laminar (Re < ~200 in the film), SI units inside the case.
Pressure in the incompressible solvers is kinematic (p / rho).
"""
from __future__ import annotations
import os, json, stat, textwrap
import numpy as np

from .foam_mesh import FoamMesh

HDR = """/*--------------------------------*- C++ -*----------------------------------*\\
  drainsponge: single hole + sponge tissue
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _w(case, rel, cls, body):
    p = os.path.join(case, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", newline="\n") as f:
        f.write(HDR.format(cls=cls, obj=os.path.basename(rel)) + textwrap.dedent(body).lstrip("\n"))


def _script(case, rel, body):
    p = os.path.join(case, rel)
    with open(p, "w", newline="\n") as f:
        f.write(textwrap.dedent(body).lstrip("\n"))
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def vector_list(V):
    return "nonuniform List<vector> \n{}\n(\n{}\n)\n".format(
        len(V), "\n".join(f"({a:.8g} {b:.8g} {c:.8g})" for a, b, c in V))


def write_case(fm: FoamMesh, case, cfg, kind="porous", iface_U=None):
    """Write mesh + dictionaries. iface_U: (n_iface_faces,3) m/s for kind='fluid'."""
    os.makedirs(case, exist_ok=True)
    fm.write(case)
    mu, rho = cfg["fluid"]["mu_Pa_s"], cfg["fluid"]["rho_kg_m3"]
    nu = mu / rho
    dp = cfg["suction"]["dp_Pa"]
    p_out = -dp / rho
    K = cfg["tissue"]["perm_m4_Ns"] * mu                 # intrinsic permeability (m^2)
    D = 1.0 / K
    of = cfg["openfoam"]
    names = [p["name"] for p in fm.patches]
    walls = [p["name"] for p in fm.patches if p["type"] == "wall"]

    # ------------------------------------------------------------- system
    sums = []
    for n in ("outlet", "film_edge", "tissue_top", "tissue_side", "tissue_iface"):
        if n in names:
            sums.append(f"""
        flow_{n}
        {{
            type            surfaceFieldValue;
            libs            ("libfieldFunctionObjects.so");
            regionType      patch;
            name            {n};
            patch           {n};
            operation       sum;
            fields          (phi);
            writeFields     false;
            writeControl    timeStep;
            writeInterval   20;
            log             false;
        }}""")
    if kind == "porous":
        sums.append("""
        flow_out_of_tissue
        {
            type            surfaceFieldValue;
            libs            ("libfieldFunctionObjects.so");
            regionType      faceZone;
            name            tissue_iface_zone;
            faceZone        tissue_iface_zone;
            operation       sum;
            fields          (phi);
            writeFields     false;
            writeControl    timeStep;
            writeInterval   20;
            log             false;
        }""")
    fo = "".join(sums)
    _w(case, "system/controlDict", "dictionary", f"""
    application     simpleFoam;            // ESI edition
    solver          incompressibleFluid;   // Foundation edition (foamRun)
    startFrom       latestTime;
    startTime       0;
    stopAt          endTime;
    endTime         {of['end_iter']};
    deltaT          1;
    writeControl    timeStep;
    writeInterval   {of['end_iter']};
    purgeWrite      1;
    writeFormat     ascii;
    writePrecision  10;
    timeFormat      general;
    runTimeModifiable true;

    functions
    {{{fo}
        wallShearStress
        {{
            type            wallShearStress;
            libs            ("libfieldFunctionObjects.so");
            patches         ({' '.join(walls)});
            writeControl    writeTime;
            log             false;
        }}
    }}
    """)
    _w(case, "system/fvSchemes", "dictionary", """
    ddtSchemes      { default steadyState; }
    gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
    divSchemes
    {
        default         none;
        div(phi,U)      bounded Gauss linearUpwind grad(U);
        div((nuEff*dev2(T(grad(U))))) Gauss linear;
        div((nuEff*dev(T(grad(U))))) Gauss linear;
    }
    laplacianSchemes { default Gauss linear limited corrected 0.5; }
    interpolationSchemes { default linear; }
    snGradSchemes   { default limited corrected 0.5; }
    wallDist        { method meshWave; }
    """)
    consistent = "no" if kind == "porous" else "yes"
    relax_p = 0.3 if kind == "porous" else 0.7
    relax_U = 0.7 if kind == "porous" else 0.9
    _w(case, "system/fvSolution", "dictionary", f"""
    solvers
    {{
        p
        {{
            solver          GAMG;
            smoother        GaussSeidel;
            tolerance       1e-9;
            relTol          0.05;
        }}
        "(U|Phi)"
        {{
            solver          smoothSolver;
            smoother        symGaussSeidel;
            tolerance       1e-10;
            relTol          0.1;
        }}
    }}
    SIMPLE
    {{
        nNonOrthogonalCorrectors 1;
        consistent      {consistent};
        residualControl {{ p 1e-6; U 1e-7; }}
    }}
    relaxationFactors
    {{
        equations {{ U {relax_U}; ".*" {relax_U}; }}
        fields    {{ p {relax_p}; }}
    }}
    """)
    _w(case, "system/decomposeParDict", "dictionary", f"""
    numberOfSubdomains {of['n_procs']};
    method          scotch;
    """)
    if kind == "porous":
        _w(case, "system/fvOptions", "dictionary", f"""
        // tissue = porous sponge: Darcy drag nu*D*U with D = 1/K, K = k*mu = {K:.4g} m^2
        tissuePorosity
        {{
            type            explicitPorositySource;
            explicitPorositySourceCoeffs
            {{
                selectionMode   cellZone;
                cellZone        tissue;
                type            DarcyForchheimer;
                d               ({D:.6g} {D:.6g} {D:.6g});
                f               (0 0 0);
                coordinateSystem
                {{
                    origin  (0 0 0);
                    e1      (1 0 0);
                    e2      (0 1 0);
                }}
            }}
        }}
        """)
    # ------------------------------------------------------------- constant
    _w(case, "constant/physicalProperties", "dictionary", f"""
    viscosityModel  constant;
    nu              [0 2 -1 0 0 0 0] {nu:.6g};
    """)
    _w(case, "constant/transportProperties", "dictionary", f"""
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] {nu:.6g};
    """)
    for n in ("momentumTransport", "turbulenceProperties"):
        _w(case, f"constant/{n}", "dictionary", "simulationType  laminar;\n")

    # ------------------------------------------------------------- 0/
    Ub, pb = {}, {}
    open_U = "type            pressureInletOutletVelocity;\n            value           uniform (0 0 0);"
    for n in names:
        if n in walls:
            Ub[n] = "type            noSlip;"
            pb[n] = "type            zeroGradient;"
        elif n == "outlet":
            Ub[n] = open_U
            pb[n] = f"type            fixedValue;\n            value           uniform {p_out:.8g};"
        elif n == "film_edge":                      # film open to the surrounding liquid at p = 0
            Ub[n] = open_U
            pb[n] = "type            totalPressure;\n            p0              uniform 0;\n            value           uniform 0;"
        elif n in ("tissue_top", "tissue_side"):    # liquid inside the organ at body pressure 0
            Ub[n] = open_U
            pb[n] = "type            fixedValue;\n            value           uniform 0;"
        elif n == "tissue_iface":                   # moving / seeping tissue surface (from FEBio)
            val = "uniform (0 0 0)" if iface_U is None else vector_list(iface_U)
            Ub[n] = f"type            fixedValue;\n            value           {val};"
            pb[n] = "type            zeroGradient;"
    body = lambda B: "\n".join(f"    {n}\n    {{\n            {B[n]}\n    }}" for n in names)
    _w(case, "0/U", "volVectorField", f"""
    dimensions      [0 1 -1 0 0 0 0];
    internalField   uniform (0 0 0);
    boundaryField
    {{
{body(Ub)}
    }}
    """)
    _w(case, "0/p", "volScalarField", f"""
    dimensions      [0 2 -2 0 0 0 0];
    internalField   uniform 0;
    boundaryField
    {{
{body(pb)}
    }}
    """)
    # ------------------------------------------------------------- scripts
    need_esi = "1" if kind == "porous" else "0"
    _script(case, "Allrun", f"""
    #!/bin/bash
    # Needs a sourced OpenFOAM environment. NP = MPI ranks (default {of['n_procs']}).
    set -e
    cd "$(dirname "$0")"
    NP=${{NP:-{of['n_procs']}}}
    if command -v simpleFoam >/dev/null 2>&1; then SOLVER="simpleFoam"
    elif [ "{need_esi}" = "1" ]; then
        echo "The porous (Darcy-Brinkman) case uses fvOptions -> needs the ESI edition (openfoam.com, simpleFoam)."; exit 1
    else SOLVER="foamRun"; fi
    echo "solver: $SOLVER  ranks: $NP"
    checkMesh > log.checkMesh 2>&1 || echo "checkMesh reported problems - see log.checkMesh"
    if [ "$NP" -gt 1 ]; then
        decomposePar -force > log.decomposePar 2>&1
        mpirun -np $NP $SOLVER -parallel > log.solver 2>&1
        reconstructPar -latestTime > log.reconstructPar 2>&1
        rm -rf processor*
    else
        $SOLVER > log.solver 2>&1
    fi
    foamToVTK -latestTime > log.foamToVTK 2>&1 || true
    touch case.foam
    echo "done - flow rates in postProcessing/, open case.foam in ParaView"
    """)
    _script(case, "Allclean", """
    #!/bin/bash
    cd "$(dirname "$0")"
    rm -rf processor* postProcessing VTK log.* [1-9]* 0.[0-9]*
    """)
    info = dict(kind=kind, nu=nu, mu=mu, rho=rho, dp_Pa=dp, p_out_kinematic=p_out,
                K_m2=K, darcy_D=D, patches=fm.patches, zones={z: int(len(v)) for z, v in fm.zones.items()})
    json.dump(info, open(os.path.join(case, "case_info.json"), "w"), indent=1)
    return case


# ------------------------------------------------------------------ reading results
def latest_time(case):
    ts = []
    for d in os.listdir(case):
        try:
            if float(d) > 0 and os.path.isdir(os.path.join(case, d)):
                ts.append((float(d), d))
        except ValueError:
            pass
    return max(ts)[1] if ts else None


def read_internal_scalar(path):
    """internalField of an ascii volScalarField."""
    txt = open(path).read()
    i = txt.index("internalField")
    head = txt[i:i + 200]
    if "nonuniform" not in head:
        v = float(head.split("uniform")[1].split(";")[0])
        return v
    j = txt.index("(", i)
    n = int(txt[txt.index(">", i) + 1:j].split()[-1])
    k = txt.index(")", j)
    return np.array(txt[j + 1:k].split(), float)[:n]


def iface_pressure(fm: FoamMesh, case, rho):
    """Static pressure (Pa) on tissue_iface faces = owner-cell value (zeroGradient)."""
    t = latest_time(case)
    p = read_internal_scalar(os.path.join(case, t, "p"))
    faces = fm.patch_faces("tissue_iface")
    return rho * np.asarray(p)[fm.owner[faces]]

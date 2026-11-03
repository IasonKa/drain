"""Fixed-level OpenFOAM run of one case, light enough for a laptop or desktop (Linux or Windows).

    python3 vof/local/make_local_case.py rec_tilt10 8.0 [--end 0.12] [--write 0.01] [--air 1]

Copies the case template vof/cases/<case> to vof/local/runs/<case>_h<level>/ and changes it from a
drainage run (pool starts above the drain and empties for seconds) to a snapshot at one pool level:
- mesh: same levels as 'coarse' (skin 0.4 mm, opening rims, caps and bore 0.2 mm), but the drain
  interior at 0.8 mm away from the skin and the pool at the 1.6 mm background cell;
- start: pool liquid below the level h, liquid inside the drain below the model's inner level
  zeta(h), air above; the run lasts about 0.1 s, long enough for the air and liquid flows through
  the openings and the bore to settle, short enough that the pool level hardly changes;
- output: all fields every --write seconds for the pictures, plus the monitors of the template.
model_state.json in the run holds the model's prediction at that level for the comparison.
"""
import argparse
import json
import math
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOF = os.path.dirname(HERE)
ROOT = os.path.dirname(VOF)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_study  # noqa: E402

TILT = 10.0
SPECS = {"usz_tilt10": dict(layout="usz", d=1.0, family="reference", tilt=TILT),
         "rec_tilt10": None}                                  # recommended.json, filled in below


def model_state(case, h):
    spec = SPECS[case] or dict(run_study.best_spec(), tilt=TILT)
    m = run_study.make_model(spec)
    br = np.zeros(m.pattern.n, bool)
    for hh in np.arange(m.h_start(), h, -0.25):              # march down so openings breach in order
        br = m.state(hh + m.Z_min, br)["breached"]
    st = m.state(h + m.Z_min, br)
    return dict(case=case, tilt=TILT, h=h, mode=st["mode"], zeta=float(st["zeta"] - m.Z_min),
                pg=float(st["pg"]), qL_ml_min=st["qL"] * 6e7, qG_l_min=st["qG"] * 6e4,
                n_open=int(st["breached"].sum()), z_lip=float(m.Z_lip - m.Z_min),
                h_stop=float(m.descend(dh=0.1)["h_stop"]))


def rotated_box(level_mm, sa, ca, below=0.08, width=0.06, length=0.18, left=-0.03):
    """rotatedBoxToCell entries (drain frame, m) for everything below the world level."""
    lv = level_mm * 1e-3
    ex = np.array([ca, 0.0, -sa])                  # world horizontal, in the drain frame
    ez = np.array([sa, 0.0, ca])                   # world up
    origin = left * ex + (lv - below) * ez + np.array([0.0, -width / 2, 0.0])

    def v(a):
        return "(" + " ".join(f"{c:.9g}" for c in a) + ")"
    return (f"origin  {v(origin)};\n        i       {v(length * ex)};\n"
            f"        j       {v([0, width, 0])};\n        k       {v(below * ez)};")


VTK_ASCII = "formatOptions { vtk { format ascii; legacy true; } }"


def slices(case, level, sa, ca):
    """Sampled planes and the drain surface, written with every field write (for the pictures).

    midPlane: y = 0, through the drain axis and the bore.
    rowPlane: through a row (column) of openings: the -30 deg row (recommended), or the vertical
      plane through the bottom and top openings at y = a sin(-5.1 deg) (USZ, drilled through both faces).
    xLiquid / xAir: cross-sections through a submerged and an exposed opening of that row.
    """
    spec = SPECS[case] or dict(run_study.best_spec(), tilt=TILT)
    m = run_study.make_model(spec)
    pat, a, b = m.pattern, m.drain.a, m.drain.b
    th_row = math.radians(-30.0) if case == "rec_tilt10" else pat.theta[np.argmin(np.abs(pat.theta + 0.089))]
    yr, zr = a * math.sin(th_row), b * (1 - math.cos(th_row))
    if case == "rec_tilt10":
        n_row = (0.0, -(zr - b), yr)                       # plane through the axis and the row
        p_row = (0.0, 0.0, b)
    else:
        n_row, p_row = (0.0, 1.0, 0.0), (0.0, yr, 0.0)
    if case == "rec_tilt10":
        on_row = np.isclose(pat.theta, th_row)
    else:                                                  # bottom and top openings of that column
        on_row = np.isclose(a * np.sin(pat.theta), yr, atol=0.05)
    zl = b * (1 - np.cos(pat.theta))
    Zw = pat.x * sa + zl * ca
    sub = on_row & (Zw < level - 1.5)
    exp_ = on_row & (Zw > level + 1.5)
    x_liq = float(np.median(pat.x[sub])) if sub.any() else 20.0
    x_air = float(np.median(pat.x[exp_])) if exp_.any() else 100.0
    x_liq = float(pat.x[sub][np.argmin(np.abs(pat.x[sub] - x_liq))]) if sub.any() else x_liq
    x_air = float(pat.x[exp_][np.argmin(np.abs(pat.x[exp_] - x_air))]) if exp_.any() else x_air

    def plane(name, point, normal):
        p = " ".join(f"{c * 1e-3:.6g}" for c in point)
        n = " ".join(f"{c:.6g}" for c in normal)
        return (f"            {name} {{ type cuttingPlane; planeType pointAndNormal;\n"
                f"                pointAndNormalDict {{ point ({p}); normal ({n}); }} interpolate false; }}\n")
    return ("    // planes and the drain surface for the pictures (vof/local/render_cfd.py)\n"
            "    slices\n    {\n        type            surfaces;\n        libs            (\"libsampling.so\");\n"
            "        writeControl    writeTime;\n        surfaceFormat   vtk;\n        " + VTK_ASCII + "\n"
            "        fields          (alpha.liquid p p_rgh U);\n        interpolationScheme cell;\n"
            "        surfaces\n        {\n"
            + plane("midPlane", (0, 0, 0), (0, 1, 0))
            + plane("rowPlane", p_row, n_row)
            + plane("xLiquid", (x_liq, 0, 0), (1, 0, 0))
            + plane("xAir", (x_air, 0, 0), (1, 0, 0))
            + "            drain { type patch; patches (drain); interpolate false; }\n"
            "        }\n    }\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(SPECS))
    ap.add_argument("level", type=float, help="pool level h (mm above the lowest point of the skin)")
    ap.add_argument("--end", type=float, default=0.12, help="s of simulated time")
    ap.add_argument("--write", type=float, default=0.01, help="s between field writes")
    ap.add_argument("--air", type=float, default=1.0, help="air density factor (1 = real air)")
    ap.add_argument("--out", default=os.path.join(HERE, "runs"))
    a = ap.parse_args()

    src = os.path.join(VOF, "cases", a.case)
    if not os.path.exists(os.path.join(src, "constant", "triSurface", "drain.stl")):
        sys.exit(f"missing STL surfaces: run  py -3.12 vof/make_cases.py --no-estimate --only {a.case}")
    name = f"{a.case}_h{a.level:g}" + (f"_air{a.air:g}" if a.air != 1 else "")
    dst = os.path.join(a.out, name)
    if os.path.exists(dst):                        # empty it (the folder itself may be in use on Windows)
        for e in os.scandir(dst):
            shutil.rmtree(e.path) if e.is_dir() else os.remove(e.path)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("snappyHexMeshDict", "setFieldsDict"),
                    dirs_exist_ok=True)
    al = math.radians(TILT)
    sa, ca = math.sin(al), math.cos(al)

    st = model_state(a.case, a.level)
    json.dump(st, open(os.path.join(dst, "model_state.json"), "w"), indent=1)
    print(f"{name}: model zeta {st['zeta']:.2f} mm, pg {st['pg']:.1f} Pa, liquid {st['qL_ml_min']:.0f} mL/min, "
          f"air {st['qG_l_min']:.1f} L/min, {st['n_open']} openings open to air")

    # mesh: interior and pool one level coarser than 'coarse'
    shm = open(os.path.join(src, "system", "snappyHexMeshDict.coarse")).read()
    shm = re.sub(r"(lumen\s*\{ mode inside; levels \(\(1e15) 2\)\); \}", r"\1 1)); }", shm)
    shm = re.sub(r"\n\s*poolRegion\s*\{ mode inside;[^\n]*", "", shm)
    shm = re.sub(r"// coarse:[^\n]*\n//[^\n]*\n",
                 "// local (vof/local/make_local_case.py): skin 0.4 mm, openings/caps 0.2 mm, bore 0.2 mm,\n"
                 "//        lumen 0.8 mm, pool and air 1.6 mm\n", shm)
    open(os.path.join(dst, "system", "snappyHexMeshDict"), "w", newline="\n").write(shm)

    # initial liquid: pool below h, drain interior and tube below the model's inner level
    zeta = st["zeta"] if math.isfinite(st["zeta"]) and st["mode"] != 1 else 60.0
    topo = open(os.path.join(src, "system", "topoSetDict")).read()
    extra = f"""    {{
        name    lumenLiquid;
        type    cellSet;
        action  new;
        source  zoneToCell;
        zones   (lumen);
    }}
    {{
        name    lumenLiquid;
        type    cellSet;
        action  subset;
        source  rotatedBoxToCell;
        {rotated_box(zeta, sa, ca)}
    }}
);"""
    topo = topo[: topo.rstrip().rfind(");")] + extra + "\n"
    open(os.path.join(dst, "system", "topoSetDict"), "w", newline="\n").write(topo)
    sf = open(os.path.join(src, "system", "setFieldsDict.short")).read()
    head = sf[: sf.index("defaultFieldValues")]
    head = re.sub(r"// short start:.*?\n(//[^\n]*\n)*", "", head, flags=re.S)
    body = f"""// fixed-level start: pool liquid below h = {a.level:g} mm; inside the drain and the tube below the
// model's inner level zeta = {zeta:.2f} mm (world heights above the lowest point of the skin)
defaultFieldValues
(
    volScalarFieldValue alpha.liquid 0
);
regions
(
    rotatedBoxToCell
    {{
        {rotated_box(a.level, sa, ca)}
        fieldValues
        (
            volScalarFieldValue alpha.liquid 1
        );
    }}
    cellToCell
    {{
        sets    (lumenLiquid);
        fieldValues
        (
            volScalarFieldValue alpha.liquid 1
        );
    }}
);
"""
    open(os.path.join(dst, "system", "setFieldsDict"), "w", newline="\n").write(head + body)

    # control: short run, frequent writes, tube section closer to the bore kept as in the template
    cd = open(os.path.join(src, "system", "controlDict")).read()
    env = dict(re.findall(r"^(\w+)=([^\s#]+)", open(os.path.join(src, "case.env")).read(), re.M))
    cd = re.sub(r"^endTime\s+[^;]+;.*$", f"endTime         {a.end:g};", cd, flags=re.M)
    cd = re.sub(r"^writeInterval\s+[^;]+;.*$", f"writeInterval   {a.write:g};", cd, flags=re.M)
    cd = re.sub(r"^maxDeltaT\s+[^;]+;.*$", f"maxDeltaT       {env['MAX_DT_coarse']};", cd, flags=re.M)
    cd = re.sub(r"^startFrom\s+[^;]+;", "startFrom       latestTime;", cd, flags=re.M)
    cd = re.sub(r"(freeSurface\s*\{[^}]*?)writeControl\s+runTime;\s*writeInterval\s+[^;]+;\s*surfaceFormat\s+vtk;",
                rf"\1writeControl    writeTime;\n        surfaceFormat   vtk;\n        {VTK_ASCII}", cd, flags=re.S)
    cd = re.sub(r"^writeFormat\s+[^;]+;", "writeFormat     ascii;", cd, flags=re.M)   # read by render_cfd.py
    cd = cd[: cd.rstrip().rfind("}")] + slices(a.case, a.level, sa, ca) + "}\n"
    open(os.path.join(dst, "system", "controlDict"), "w", newline="\n").write(cd)
    if a.air != 1:
        tp = open(os.path.join(dst, "constant", "transportProperties")).read()
        rho, mu = float(env["RHO_AIR"]), float(env["MU_AIR"])
        tp = re.sub(r"(air\s*\{[^}]*?rho\s+)[^;]+;", rf"\g<1>{rho * a.air:.6g};", tp, flags=re.S)
        tp = re.sub(r"(air\s*\{[^}]*?nu\s+)[^;]+;", rf"\g<1>{mu / (rho * a.air):.6g};", tp, flags=re.S)
        open(os.path.join(dst, "constant", "transportProperties"), "w", newline="\n").write(tp)
    for f in ("Allmesh", "Allclean", "case.env"):
        p = os.path.join(dst, f)
        if os.path.exists(p):
            os.remove(p)
    shutil.copy(os.path.join(HERE, "Allrun.local"), os.path.join(dst, "Allrun"))
    print("wrote", dst)


if __name__ == "__main__":
    main()

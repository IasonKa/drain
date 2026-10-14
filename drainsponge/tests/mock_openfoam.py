"""
Stand-in for the OpenFOAM Allrun used by tests/test_coupling.py: reads the mesh size from
constant/polyMesh/owner and the tissue_iface velocity from 0/U, and writes a solution
time directory "100" with a plausible kinematic pressure (suction falling off with
distance from the hole axis) in OpenFOAM's ascii format (gzip if compression is on).
"""
import gzip, os, re, sys
import numpy as np

case = sys.argv[1] if len(sys.argv) > 1 else "."
owner = open(os.path.join(case, "constant/polyMesh/owner")).read()
nC = int(re.search(r"nCells:\s*(\d+)", owner).group(1))
U0 = os.path.join(case, "0", "U")
txt = open(U0).read() if os.path.exists(U0) else gzip.open(U0 + ".gz", "rt").read()
assert "tissue_iface" in txt
pts = open(os.path.join(case, "constant/polyMesh/points")).read()
P = np.array(re.findall(r"\(([-0-9.eE+]+) ([-0-9.eE+]+) ([-0-9.eE+]+)\)", pts), float)
faces = open(os.path.join(case, "constant/polyMesh/faces")).read()
F = np.array(re.findall(r"4\((\d+) (\d+) (\d+) (\d+)\)", faces), int)
own = np.array(owner[owner.index("(") + 1:owner.rindex(")")].split(), int)
cc = np.zeros((nC, 3)); cnt = np.zeros(nC)
fc = P[F].mean(1)
np.add.at(cc, own, fc); np.add.at(cnt, own, 1)
cc /= cnt[:, None]
r = np.hypot(cc[:, 0], cc[:, 1]) * 1e3                 # mm
p = -4.9 * np.exp(-((r / 0.6) ** 2))                    # kinematic
comp = "writeCompression on" in open(os.path.join(case, "system/controlDict")).read()
os.makedirs(os.path.join(case, "100"), exist_ok=True)
p0 = open(os.path.join(case, "0", "p")).read() if os.path.exists(os.path.join(case, "0", "p")) else gzip.open(os.path.join(case, "0", "p.gz"), "rt").read()
m = re.search(r"outlet\s*\{[^{}]*uniform\s+([-0-9.eE+]+)", p0)
assert m, "outlet pressure missing in 0/p"
p *= float(m.group(1)) / -4.9 if float(m.group(1)) != 0 else 0.0      # follows the (ramped) suction
body_p = ("FoamFile{}\ninternalField   nonuniform List<scalar> \n%d\n(\n%s\n)\n;\nboundaryField\n{\n"
          "    outlet\n    {\n        type            fixedValue;\n        value           uniform %s;\n    }\n}\n"
          % (nC, "\n".join(f"{v:.6g}" for v in p), m.group(1)))
body_U = "FoamFile{}\ninternalField   nonuniform List<vector> \n%d\n(\n%s\n)\n;\nboundaryField\n{\n    tissue_iface\n    {\n        type fixedValue;\n        value uniform (0 0 0);\n    }\n}\n" % (nC, "\n".join("(0 0 0)" for _ in range(nC)))
for name, body in (("p", body_p), ("U", body_U)):
    path = os.path.join(case, "100", name)
    if comp:
        with gzip.open(path + ".gz", "wt") as f:
            f.write(body)
    else:
        open(path, "w").write(body)
print("mock OpenFOAM done")

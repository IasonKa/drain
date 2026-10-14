"""
Stand-in for febio4 used by tests/test_coupling.py. It reads the same input/restart files
the coupler writes, keeps a fake state in the dump, and writes logs in FEBio's format.
"Physics": the tissue bottom moves down by w = p * compliance (bounded by the gap) and
liquid seeps out at a rate proportional to -p. Only for checking the plumbing.

usage: python mock_febio.py -i tissue.feb -dump=state.dmp -silent
       python mock_febio.py -r restart.feb -dump=state.dmp -silent
"""
import sys, os, json, re
import xml.etree.ElementTree as ET
import numpy as np

FAIL_RESTART = os.environ.get("MOCK_FEBIO_FAIL_RESTART") == "1"


def curves(root_loaddata):
    out = {}
    for lc in root_loaddata.findall("load_controller"):
        pts = [tuple(map(float, p.text.split(","))) for p in lc.find("points").findall("pt")]
        out[int(lc.get("id"))] = pts
    return out


def steps_end(step_root, t0):
    t = t0
    for st in step_root.findall("step"):
        c = st.find("Control")
        t += int(c.find("time_steps").text) * float(c.find("step_size").text)
    return t


def evalc(pts, t):
    ts = np.array([p[0] for p in pts]); vs = np.array([p[1] for p in pts])
    return float(np.interp(t, ts, vs))


def write_logs(state, t, append):
    mode = "a" if append else "w"
    nodes = np.array(state["nodes"]); bottom = state["bottom"]; patch_of = state["patch_of_node"]
    lc = {int(k): v for k, v in state["curves"].items()}
    p = np.array([evalc(lc[patch_of[str(n)]], t) if str(n) in patch_of else 0.0 for n in bottom])   # MPa
    dz = -np.clip(-p * 40.0, 0.0, 0.3)                                    # mm, down when p < 0 (suction)
    with open("bottom_xyz.txt", mode) as f:
        f.write(f"*Step  = 1\n*Time  = {t:.10g}\n*Data  = x;y;z\n")
        for n, d in zip(bottom, dz):
            x, y, z = nodes[n - 1]
            f.write(f"{n},{x},{y},{z + d}\n")
    with open("bottom_flux.txt", mode) as f:
        f.write(f"*Step  = 1\n*Time  = {t:.10g}\n*Data  = wx;wy;wz\n")
        for e in state["bottom_elems"]:
            f.write(f"{e},0,0,{1e-6}\n")
    with open("tissue_state.txt", mode) as f:
        f.write(f"*Step  = 1\n*Time  = {t:.10g}\n*Data  = ux;uy;uz;pe\n")
        for i in range(len(nodes)):
            f.write(f"{i+1},0,0,0,0\n")


def main():
    a = sys.argv[1:]
    dump = [x.split("=", 1)[1] for x in a if x.startswith("-dump=")][0]
    if "-i" in a:
        root = ET.parse(a[a.index("-i") + 1]).getroot()
        mesh = root.find("Mesh")
        nodes = [list(map(float, n.text.split(","))) for n in mesh.find("Nodes[@name='TissueNodes']")]
        sets = {ns.get("name"): [int(v) for v in ns.text.split(",")] for ns in mesh.findall("NodeSet") if ns.text}
        bottom = sets["Bottom"]
        patch_of = {}
        for name, ids in sets.items():
            if re.fullmatch(r"pn\d+", name):
                for n in ids:
                    patch_of[str(n)] = int(name[2:])
        belems = [int(v) for v in mesh.find("ElementSet[@name='BottomElems']").text.split(",")]
        state = dict(nodes=nodes, bottom=bottom, patch_of_node=patch_of, bottom_elems=belems,
                     curves={str(k): v for k, v in curves(root.find("LoadData")).items()})
        t = steps_end(root.find("Step"), 0.0)
        state["t"] = t
        write_logs(state, t, append=False)
    else:
        if FAIL_RESTART:
            print("mock: restart not supported"); sys.exit(1)
        root = ET.parse(a[a.index("-r") + 1]).getroot()
        arch = root.find("Archive").text.strip()
        state = json.load(open(arch))
        for k, v in curves(root.find("LoadData")).items():
            state["curves"][str(k)] = v
        t = steps_end(root.find("Step"), state["t"])
        state["t"] = t
        write_logs(state, t, append=True)
    json.dump(state, open(dump, "w"))
    open("tissue.log", "a").write(" N O R M A L   T E R M I N A T I O N\n")


if __name__ == "__main__":
    main()

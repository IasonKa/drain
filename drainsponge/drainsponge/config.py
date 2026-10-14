"""Default parameters for the sponge-tissue / single-hole study (all in one place).

    cfg = config.load(path=None, design="standoff", resolution="default", **section_overrides)

design      : "current"  - drainage-3D dome-cap design, sharp Ø1.0 hole, tissue 50 um above the skin
              "standoff" - new offset design: ring r 0.55..0.85, 0.40 mm high, 4 slots 0.35 mm,
                           tissue 50 um above the ring tops
              "standoff_opt" - screening result: ring r 0.55..1.00, 0.50 mm high, 4 slots 0.50 mm
resolution  : "test" (minutes, for checking a cluster setup), "default" (laptop / small node),
              "fine" (Euler production: standoff ~0.25 M fluid cells, ~0.22 M tissue hexes = ~0.9 M
                      biphasic unknowns; current design ~0.14 M / 0.17 M)
"""
import json, copy

DESIGNS = {
    "current": dict(standoff_h=0.0),
    "standoff": dict(standoff_h=0.40, standoff_rin=0.55, standoff_rout=0.85,
                     slot_w=0.35, n_slots=4, slot_angle0=0.0),
    # best of the preview screening (scripts/optimize_standoff.py): lowest ring keeping the tissue
    # >= 100 um above the hole rim for E = 10 kPa at 10 kPa suction with no free liquid
    "standoff_opt": dict(standoff_h=0.50, standoff_rin=0.55, standoff_rout=1.00,
                         slot_w=0.50, n_slots=4, slot_angle0=0.0),
}

RESOLUTIONS = {
    "test": dict(nq=8, n_in=3, n_gap=1, n_ring=3, n_out=8, n_chan=4, n_wall=5, n_stand=4, n_film=3,
                 n_tis=10, rim_cell=0.04, tis_cell=0.06),
    "default": dict(),
    "fine": dict(nq=28, n_in=9, n_gap=3, n_ring=9, n_out=32, n_chan=12, n_wall=16, n_stand=14,
                 n_film=8, n_tis=32, rim_cell=0.008, tis_cell=0.01),
}

DEFAULTS = {
    "design": "standoff",
    "resolution": "default",
    "window": {},                          # geom.Window fields (override design/resolution presets)
    "fluid": {
        "mu_Pa_s": 1.5e-3,                 # wound fluid / seroma (placeholder; blood ~3.5e-3)
        "rho_kg_m3": 1020.0,
    },
    "suction": {
        "dp_Pa": 5000.0,                   # suction in the drain channel (~ -38 mmHg). Placeholder.
        "ramp_s": 0.01,                    # suction switched on over this time
    },
    "tissue": {                            # biphasic sponge (placeholders for soft wound tissue)
        "E_Pa": 20000.0,                   # drained Young's modulus of the solid matrix
        "nu": 0.2,                         # drained Poisson's ratio
        "phi0": 0.2,                       # solid volume fraction (80 % liquid)
        "perm_m4_Ns": 1e-14,               # hydraulic permeability k (Darcy: w = -k grad p)
        "perm_M": 0.0,                     # Holmes-Mow strain dependence exponent (0 = constant)
        "perm_alpha": 0.0,
        "friction": 0.2,                   # tissue / plastic contact
    },
    "time": {
        "t_end_s": 3600.0,                 # simulated time after suction on
        "n_windows": 24,                   # coupling windows (log spaced)
        "t_first_s": 1e-3,
        "febio_steps_per_window": 6,
    },
    "coupling": {
        "max_iter": 25,
        "min_iter": 2,
        "tol_disp_mm": 1e-3,
        "tol_p_rel": 1e-3,
        "omega0": 0.3,                     # initial Aitken relaxation
        "n_patches": 600,                  # FEBio interface pressure patches (load curves)
        "febio_restart": True,             # continue each window from the last FEBio dump
        "save_every_window": True,         # interface + VTU snapshot after every converged window
    },
    "openfoam": {
        "n_procs": 4,
        "end_iter": 3000,
        "write_compression": True,
    },
    "febio": {
        "exe": "febio4",
        "threads": 4,
    },
}

RES_COUPLING = {"test": dict(n_patches=120), "default": dict(), "fine": dict(n_patches=1500)}


def load(path=None, design=None, resolution=None, **over):
    cfg = copy.deepcopy(DEFAULTS)
    user = json.load(open(path)) if path else {}
    for k, v in user.items():
        if isinstance(v, dict):
            cfg.setdefault(k, {}).update(v)
        else:
            cfg[k] = v
    if design:
        cfg["design"] = design
    if resolution:
        cfg["resolution"] = resolution
    for k, v in over.items():
        if isinstance(v, dict):
            cfg.setdefault(k, {}).update(v)
        else:
            cfg[k] = v
    # window = design preset + resolution preset + explicit window entries
    win = {}
    win.update(DESIGNS[cfg["design"]])
    win.update(RESOLUTIONS[cfg["resolution"]])
    win.update(user.get("window", {}))
    win.update(over.get("window", {}))
    cfg["window"] = win
    for k, v in RES_COUPLING[cfg["resolution"]].items():
        if k not in user.get("coupling", {}) and k not in over.get("coupling", {}):
            cfg["coupling"][k] = v
    return cfg

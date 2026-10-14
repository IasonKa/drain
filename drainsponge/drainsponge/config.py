"""Default parameters for the sponge-tissue / single-hole study (all in one place)."""
import json, copy

DEFAULTS = {
    "window": {},                          # geom.Window fields (defaults = current drain design)
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
        "t_end_s": 3600.0,                 # one hour of drainage
        "n_windows": 24,                   # coupling windows (log spaced)
        "t_first_s": 1e-3,
    },
    "coupling": {
        "max_iter": 25,
        "tol_disp_mm": 1e-3,
        "tol_p_rel": 1e-3,
        "omega0": 0.3,                     # initial Aitken relaxation
        "n_patches_ring": 10,              # FEBio interface load patches: rings x sectors
        "n_patches_sector": 16,
    },
    "openfoam": {
        "n_procs": 4,
        "end_iter": 3000,
        "edition": "auto",                 # auto | esi | org
    },
    "febio": {
        "exe": "febio4",
    },
}


def load(path=None, **over):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        user = json.load(open(path))
        for k, v in user.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
    for k, v in over.items():
        cfg[k].update(v)
    return cfg

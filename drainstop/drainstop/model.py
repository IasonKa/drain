"""Reduced-order hydraulics of a perforated suction drain lying in a pool.

The drain is the USZ-scale elliptical body (outer 17.85 x 9.67 mm, 119 mm long,
0.6 mm wall) lying flat, with a closed tip at x = 0 and a centred outlet bore
(4.24 mm) in the end face at x = L. A regulated suction p_s acts in the outlet
tube just downstream of the bore entrance.

At every pool level H the model solves a steady network:

    pool  --(submerged parts of the openings, liquid)-->  drain interior
    air   --(exposed parts of the openings, gas)------->  drain interior
    drain interior --(outlet bore, gas above / liquid below the inner level)--> suction

Geometry is entered in mm. Hydraulics are in SI (Pa, m^3/s). Heights z are
"world" heights in mm (vertical), measured from the drain axis frame, and
pressures are gauge pressures (atmosphere = 0).
"""
from dataclasses import dataclass, field
import math
import numpy as np

G = 9.81
P_ATM = 101325.0
MMHG = 133.322


# ----------------------------------------------------------------------------- parameters
@dataclass
class Fluid:
    name: str = "50 wt% glycerol-water, 20 C"
    rho: float = 1126.0      # kg/m3
    mu: float = 6.0e-3       # Pa s
    gamma: float = 0.068     # N/m


FLUIDS = {
    "water": Fluid("water, 20 C", 998.0, 1.0e-3, 0.072),
    "gly40": Fluid("40 wt% glycerol-water (blood analogue), 20 C", 1099.0, 3.7e-3, 0.069),
    "gly50": Fluid(),
    "gly80": Fluid("80 wt% glycerol-water, 20 C", 1208.0, 60.0e-3, 0.065),
}


@dataclass
class Gas:
    rho: float = 1.204       # kg/m3, air at 20 C
    mu: float = 1.81e-5      # Pa s


@dataclass
class Drain:
    a: float = 8.925         # mm, outer lateral semi-axis (17.85 mm wide)
    b: float = 4.835         # mm, outer vertical semi-axis (9.67 mm high, lying flat)
    length: float = 119.0    # mm, tip (x = 0) to outlet end face (x = L)
    wall: float = 0.6        # mm, skin thickness = hole length
    port_d: float = 4.24     # mm, outlet bore
    x_lo: float = 5.0        # mm, perforated span
    x_hi: float = 113.0

    @property
    def port_area(self):     # m2
        return math.pi * (self.port_d * 1e-3) ** 2 / 4


@dataclass
class Suction:
    p_set: float = -7.0 * MMHG   # Pa gauge, regulated in the outlet tube
    q_max: float = math.inf      # m3/s, source capacity (volume at the source); inf = ideal regulator


@dataclass
class Options:
    theta_e: float = 90.0        # deg, effective contact angle for capillary hold of exposed holes
    kappa: float = 0.0           # drawdown margin: a part counts as exposed above H - kappa*d
    floor_gap: float = 0.0       # mm, opening parts closer than this to the floor are blocked
    cd_port: float = 0.85        # outlet bore entrance (sudden contraction, static p downstream)
    cd_scale: float = 1.0        # multiplies the hole discharge coefficients (sensitivity)
    q_stop: float = 0.1e-6 / 60  # m3/s, "drainage has stopped" threshold (0.1 mL/min)
    n_el: int = 61               # area elements per opening
    pool_area: float = 0.016     # m2, pool footprint (200 x 80 mm) for drain times


# ----------------------------------------------------------------------------- geometry
class Ellipse:
    """Outer skin: y = a sin(theta), z = b (1 - cos(theta)); theta = 0 is the bottom line."""

    def __init__(self, a, b, n=20001):
        th = np.linspace(-np.pi, np.pi, n)
        ds = np.sqrt((a * np.cos(th)) ** 2 + (b * np.sin(th)) ** 2)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(th))])
        s -= np.interp(0.0, th, s)
        self.a, self.b, self._th, self._s = a, b, th, s
        self.perimeter = s[-1] - s[0]

    def theta(self, s):
        P = self.perimeter
        s = (np.asarray(s, float) + P / 2) % P - P / 2
        return np.interp(s, self._s, self._th)

    def s_of(self, theta):
        theta = (np.asarray(theta, float) + np.pi) % (2 * np.pi) - np.pi
        return np.interp(theta, self._th, self._s)

    def yz(self, theta):
        theta = np.asarray(theta, float)
        return self.a * np.sin(theta), self.b * (1 - np.cos(theta))


def sunflower(n):
    """n equal-area points in the unit disk (Vogel spiral)."""
    k = np.arange(n) + 0.5
    r = np.sqrt(k / n)
    phi = k * math.pi * (3 - math.sqrt(5))
    return r * np.cos(phi), r * np.sin(phi)


def cd_hole(t_over_d):
    """Discharge coefficient of a short sharp-edged hole vs length/diameter (after Lichtarowicz et al. 1965)."""
    r = np.asarray(t_over_d, float)
    return np.clip(0.61 + 0.20 * (r - 0.75) / 1.25, 0.61, 0.81)


# ----------------------------------------------------------------------------- hole patterns
@dataclass
class Pattern:
    key: str
    label: str
    x: np.ndarray            # mm, opening centre along the drain (tip = 0)
    theta: np.ndarray        # rad, position around the skin (0 bottom, +pi/2 the +y edge, pi top)
    d: np.ndarray            # mm, opening diameter
    area_scale: np.ndarray   # 1 for a full opening, 0.5 for an edge notch
    rows_deg: tuple = ()
    family: str = ""

    @property
    def n(self):
        return len(self.x)

    @property
    def open_area_mm2(self):
        return float(np.sum(self.area_scale * math.pi * self.d ** 2 / 4))


# key: (label, row angles in degrees or "ring8", axial offsets as a fraction of the pitch)
LAYOUTS = {
    "side2":   ("Side rows, 3 and 9 o'clock", (90, -90), (0, 0)),
    "topbot2": ("Top and bottom rows", (180, 0), (0, 0)),
    "cross4":  ("Four rows in line", (0, 90, 180, -90), (0, 0, 0, 0)),
    "helix4":  ("Helix, four positions", (0, 90, 180, -90), (0, 0.25, 0.5, 0.75)),
    "ring8":   ("Eight rows, even around the skin", "ring8", (0,) * 8),
    "faces4":  ("Both faces at 45 deg", (45, -45, 135, -135), (0, 0, 0, 0)),
    "low3":    ("Lower band, 0 and 45 deg", (0, 45, -45), (0, 0, 0)),
    "low2":    ("Lower flanks 45 deg, in line", (45, -45), (0, 0)),
    "low2s":   ("Lower flanks 45 deg, staggered", (45, -45), (0, 0.5)),
    "bottom1": ("Bottom row, 6 o'clock", (0,), (0,)),
}
PATTERN_KEYS = list(LAYOUTS)


def row_angles(key, ell):
    rows = LAYOUTS[key][1]
    if rows == "ring8":
        return [float(np.degrees(ell.theta(i * ell.perimeter / 8))) for i in range(8)]
    return [float(r) for r in rows]


def max_stations(d, drain, ligament=0.6):
    return max(1, int((drain.x_hi - drain.x_lo) // (d + ligament)))


def rows_pattern(key, d, n_x, drain=None, family=""):
    """Uniform layout: n_x equally spaced stations per row, every row at a fixed angle."""
    drain = drain or Drain()
    ell = Ellipse(drain.a, drain.b)
    label, _, offsets = LAYOUTS[key]
    angles = row_angles(key, ell)
    n_x = int(min(n_x, max_stations(d, drain)))
    pitch = (drain.x_hi - drain.x_lo) / n_x
    xs, ths = [], []
    for ang, off in zip(angles, offsets):
        x = drain.x_lo + (((np.arange(n_x) + 0.5 + off) % n_x)) * pitch
        xs.append(np.sort(x))
        ths.append(np.full(n_x, math.radians(ang)))
    x = np.concatenate(xs)
    th = np.concatenate(ths)
    return Pattern(key, label, x, th, np.full(x.size, float(d)), np.ones(x.size),
                   tuple(round(a, 1) for a in angles), family)


def fixed_area_pattern(key, d, area_mm2, drain=None):
    """Same layout, number of stations chosen so that the total open area is about area_mm2."""
    drain = drain or Drain()
    n_rows = len(row_angles(key, Ellipse(drain.a, drain.b)))
    n_x = max(1, int(round(area_mm2 / (n_rows * math.pi * d ** 2 / 4))))
    return rows_pattern(key, d, n_x, drain, family="fixed_area")


def usz_as_built(drain=None, d=1.0):
    """The original USZ hole field: 318 holes of 1.0 mm drilled straight through both faces.

    Axial pitch 4.5 mm with staggered rows every 2.25 mm and 12 columns across the drain
    (read from 03_figures/02_hole_pattern_map.png). A column outside the skin (|y| > a)
    opens as a half-moon notch in the side edge and is counted as half an opening.
    """
    drain = drain or Drain()
    a = drain.a
    sets = [(1.1 + 4.5 * np.arange(27), [9.1, 5.8, 2.5, -0.8, -4.1, -7.4]),
            (3.35 + 4.5 * np.arange(26), [8.3, 5.0, 1.7, -1.6, -4.9, -8.2])]
    xs, ths, sc = [], [], []
    for xcol, ys in sets:
        for x_their in xcol:
            x = drain.length - x_their          # their x runs from the outlet towards the tip
            for y in ys:
                if abs(y) < a:
                    tb = math.asin(y / a)
                    for t in (tb, math.pi - tb):
                        xs.append(x); ths.append(t); sc.append(1.0)
                else:
                    xs.append(x); ths.append(math.copysign(math.pi / 2, y)); sc.append(0.5)
    ths = (np.array(ths) + np.pi) % (2 * np.pi) - np.pi
    n = len(xs)
    return Pattern("usz", "USZ as built (1.0 mm, both faces)", np.array(xs), ths,
                   np.full(n, d), np.array(sc), (), "reference")


# ----------------------------------------------------------------------------- hydraulics
def orifice(dp, R, K):
    """Flow through a short hole: dp = R q + K q|q|  (Sampson + Poiseuille + orifice)."""
    a = np.abs(dp)
    q = 2.0 * a / (R + np.sqrt(R * R + 4.0 * K * a))
    return np.sign(dp) * q


def segment_fraction(u):
    """Fraction of a circle's area below a horizontal line at u * radius from the centre."""
    u = np.clip(u, -1.0, 1.0)
    return (math.pi - np.arccos(u) + u * np.sqrt(1 - u * u)) / math.pi


def illinois(f, a, b, fa, fb, xtol, maxit=80):
    """Root of a monotone f on [a, b] (f(a), f(b) of opposite sign), Illinois false position."""
    side, c_old = 0, None
    c = a
    for _ in range(maxit):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if fc == 0.0 or (c_old is not None and abs(c - c_old) < xtol):
            return c
        c_old = c
        if fc * fb > 0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
        if abs(b - a) < xtol:
            return c
    return c


class DrainModel:
    """One drain design at one tilt. Use state(H) for a level, descend() for a full drainage."""

    def __init__(self, pattern, tilt_deg=0.0, drain=None, fluid=None, gas=None,
                 suction=None, opts=None):
        self.pattern = pattern
        self.drain = drain = drain or Drain()
        self.fluid = fluid = fluid or Fluid()
        self.gas = gas = gas or Gas()
        self.suction = suction = suction or Suction()
        self.opts = opts = opts or Options()
        self.tilt = tilt_deg
        self.ell = Ellipse(drain.a, drain.b)
        al = math.radians(tilt_deg)
        self.ca, self.sa = math.cos(al), math.sin(al)

        # area elements of every opening, laid out on the developed skin (arc length, axial)
        u, v = sunflower(opts.n_el)
        r = pattern.d[:, None] / 2
        s_c = self.ell.s_of(pattern.theta)
        s_e = s_c[:, None] + r * u[None, :]
        x_e = pattern.x[:, None] + r * v[None, :]
        z_loc = drain.b * (1 - np.cos(self.ell.theta(s_e)))
        self.Z = x_e * self.sa + z_loc * self.ca                  # mm, world height of elements
        self.blocked = z_loc < opts.floor_gap
        self.sc = opts.kappa * pattern.d                          # mm, drawdown margin

        L, b = drain.length, drain.b
        self.Z_min = min(0.0, L * self.sa)                        # lowest point of the skin
        self.Z_top = max(2 * b * self.ca, L * self.sa + 2 * b * self.ca)
        self.Z_pc = L * self.sa + b * self.ca                     # bore centre
        self.rp = drain.port_d / 2
        self.Z_lip = self.Z_pc - self.rp * self.ca                # bore lower edge ("lip")
        self.Z_ptop = self.Z_pc + self.rp * self.ca

        d = pattern.d * 1e-3
        t = drain.wall * 1e-3
        A = math.pi * d ** 2 / 4 * pattern.area_scale
        self.A = A
        self.cd = cd_hole(drain.wall / pattern.d) * opts.cd_scale
        rv = 24.0 / d ** 3 + 128.0 * t / (math.pi * d ** 4)
        self.RL0, self.KL0 = fluid.mu * rv, fluid.rho / (2 * self.cd ** 2 * A ** 2)
        self.RG0, self.KG0 = gas.mu * rv, gas.rho / (2 * self.cd ** 2 * A ** 2)
        self.Pc = np.maximum(4 * fluid.gamma * math.cos(math.radians(opts.theta_e)) / d, 0.0)
        self.Ap = drain.port_area
        self.rgl = fluid.rho * G * 1e-3                           # Pa per mm of liquid
        self.Hv = -suction.p_set / self.rgl                       # mm, suction head in liquid

    # -------------------------------------------------------------- exposure at a level
    def exposure(self, H):
        thr = H - self.sc[:, None]
        sub = (self.Z < thr) & ~self.blocked
        exp = (self.Z >= thr) & ~self.blocked
        n = self.opts.n_el
        ns, ne = sub.sum(1), exp.sum(1)
        zs = np.where(ns > 0, (self.Z * sub).sum(1) / np.maximum(ns, 1), 0.0)
        ze = np.where(ne > 0, (self.Z * exp).sum(1) / np.maximum(ne, 1), 0.0)
        return ns / n, ne / n, zs, ze

    # -------------------------------------------------------------- flows for given unknowns
    def _liquid_in(self, H, fs, zs, p_in_sub):
        ok = fs > 0
        q = np.zeros_like(fs)
        dp = self.rgl * (H - zs[ok]) - p_in_sub[ok]
        q[ok] = orifice(dp, self.RL0[ok] / fs[ok], self.KL0[ok] / fs[ok] ** 2)
        return q

    def _gas_in(self, fe, open_, p_in_exp):
        ok = (fe > 0) & open_
        q = np.zeros_like(fe)
        dp = np.maximum(-p_in_exp[ok], 0.0)
        q[ok] = orifice(dp, self.RG0[ok] / fe[ok], self.KG0[ok] / fe[ok] ** 2)
        return q

    def _leak_out(self, fe, p_in_exp, wet_inside):
        """Liquid dripping out of exposed parts whose inner side is liquid above atmospheric."""
        ok = (fe > 0) & wet_inside & (p_in_exp > 0)
        q = np.zeros_like(fe)
        q[ok] = orifice(p_in_exp[ok], self.RL0[ok] / fe[ok], self.KL0[ok] / fe[ok] ** 2)
        return q

    def _port(self, pg, zeta, ps):
        u = (zeta - self.Z_pc) / (self.rp * self.ca)
        AL = self.Ap * segment_fraction(u)
        AG = self.Ap - AL
        zL = self.Z_pc if u >= 1 else 0.5 * (min(zeta, self.Z_ptop) + self.Z_lip)
        dpl = pg - ps + self.rgl * (zeta - zL)
        qL = self.opts.cd_port * AL * math.sqrt(2 * max(dpl, 0.0) / self.fluid.rho)
        qG = self.opts.cd_port * AG * math.sqrt(2 * max(pg - ps, 0.0) / self.gas.rho)
        return qL, qG

    # -------------------------------------------------------------- the two regimes
    def _single_phase(self, H, ex, ps):
        """Interior full of liquid (no open exposed part). Unknown: pressure pc at the bore centre."""
        fs, fe, zs, ze = ex

        def p_in(z, pc):
            return pc + self.rgl * (self.Z_pc - z)

        wet = np.ones(fe.shape, bool)

        def net_in(pc):
            return (self._liquid_in(H, fs, zs, p_in(zs, pc)).sum()
                    - self._leak_out(fe, p_in(ze, pc), wet).sum())

        def f(pc):
            qout = self.opts.cd_port * self.Ap * math.sqrt(2 * max(pc - ps, 0) / self.fluid.rho)
            return net_in(pc) - qout

        if not np.any(fs > 0):
            pc = ps
        else:
            lo, hi = ps, self.rgl * (H - self.Z_min) + 50.0
            flo, fhi = f(lo), f(hi)
            pc = lo if flo <= 0 else (hi if fhi >= 0 else illinois(f, lo, hi, flo, fhi, 1e-7))
        q = max(net_in(pc), 0.0)
        dpG = -p_in(ze, pc)
        return dict(mode=1, pg=pc, zeta=np.inf, qL=float(q), qG=0.0,
                    qLp=float(q), qGp=0.0, dpG=dpG)

    def _two_phase(self, H, ex, open_, ps):
        """Gas reaches the interior. Unknowns: gas pressure pg and inner liquid level zeta."""
        fs, fe, zs, ze = ex

        def p_in(z, pg, zeta):
            return pg + self.rgl * np.maximum(zeta - z, 0.0)

        def gas_balance(zeta):
            def fg(pg):
                qg = self._gas_in(fe, open_, p_in(ze, pg, zeta)).sum()
                return qg - self._port(pg, zeta, ps)[1]
            flo = fg(ps)
            if flo <= 0:
                return ps
            fhi = fg(0.0)
            if fhi >= 0:
                return 0.0
            return illinois(fg, ps, 0.0, flo, fhi, 1e-7)

        def liq(zeta):
            pg = gas_balance(zeta)
            qin = (self._liquid_in(H, fs, zs, p_in(zs, pg, zeta)).sum()
                   - self._leak_out(fe, p_in(ze, pg, zeta), ze < zeta).sum())
            return qin - self._port(pg, zeta, ps)[0], pg

        z_lo = self.Z_lip
        z_hi = self.Z_ptop - 1e-6 * self.rp
        g_lo, pg_lo = liq(z_lo)
        if g_lo <= 0:                       # liquid cannot reach the bore lip: stalled
            zeta, pg, stalled = z_lo, pg_lo, True
        else:
            g_hi, pg_hi = liq(z_hi)
            if g_hi >= 0:
                zeta, pg = z_hi, pg_hi
            else:
                zeta = illinois(lambda z: liq(z)[0], z_lo, z_hi, g_lo, g_hi, 1e-7)
                pg = gas_balance(zeta)
            stalled = False
        qLp, qGp = self._port(pg, zeta, ps)
        if stalled:
            qLp = 0.0
            # static inner level: the liquid stands at H + |pg| / (rho g)
            zeta = min(H - pg / self.rgl, z_lo)
        qG = self._gas_in(fe, open_, p_in(ze, pg, zeta))
        dpG = -p_in(ze, pg, zeta)
        return dict(mode=2 if not stalled else 3, pg=pg, zeta=zeta, qL=float(qLp),
                    qG=float(qG.sum()), qLp=float(qLp), qGp=float(qGp), dpG=dpG)

    def _solve_at(self, H, ex, breached, ps):
        fs, fe, zs, ze = ex
        breached = breached & (fe > 0)
        while True:
            if np.any(breached):
                st = self._two_phase(H, ex, breached, ps)
            else:
                st = self._single_phase(H, ex, ps)
            new = (fe > 0) & ~breached & (st["dpG"] > self.Pc)
            if not np.any(new):
                st["breached"] = breached
                return st
            breached = breached | new

    def state(self, H, breached=None):
        """Steady state at pool level H (world mm). breached: openings already open to air."""
        ex = self.exposure(H)
        if breached is None:
            breached = np.zeros(self.pattern.n, bool)
        ps = self.suction.p_set
        st = self._solve_at(H, ex, breached, ps)
        if math.isfinite(self.suction.q_max):
            def demand(st_):
                return st_["qLp"] + st_["qGp"] * P_ATM / (P_ATM + ps_cur[0])
            ps_cur = [ps]
            if demand(st) > self.suction.q_max:
                def f(p):
                    ps_cur[0] = p
                    return demand(self._solve_at(H, ex, st["breached"], p)) - self.suction.q_max
                p = illinois(f, self.suction.p_set, -1e-3, f(self.suction.p_set), f(-1e-3), 1e-4)
                ps_cur[0] = p
                st = self._solve_at(H, ex, st["breached"], p)
                ps = p
        fs, fe, zs, ze = ex
        st["ps"] = ps
        st["A_exposed"] = float(np.sum(fe * self.A))
        st["A_submerged"] = float(np.sum(fs * self.A))
        st["Lambda_open"] = float(np.sum(fe * self.A * self.cd * st["breached"]) /
                                  (self.opts.cd_port * self.Ap))
        return st

    # -------------------------------------------------------------- full drainage
    def h_start(self):
        return self.Z_top - self.Z_min + 3.0

    def descend(self, dh=0.1, refine=True):
        """Lower the pool quasi-statically from above the drain to its lowest point.

        Returns a dict of arrays versus h (mm above the lowest point of the drain) and the stop
        level h_stop: the first level at which the liquid flow falls below opts.q_stop.
        """
        hs = np.arange(self.h_start(), -1e-9, -dh)
        breached = np.zeros(self.pattern.n, bool)
        rec = {k: [] for k in ("h", "qL", "qG", "pg", "zeta", "mode", "Lambda", "A_exp", "ps")}
        h_stop, last_ok = None, None
        for h in hs:
            st = self.state(h + self.Z_min, breached)
            breached = st["breached"]
            for k, v in (("h", h), ("qL", st["qL"]), ("qG", st["qG"]), ("pg", st["pg"]),
                         ("zeta", st["zeta"] - self.Z_min), ("mode", st["mode"]),
                         ("Lambda", st["Lambda_open"]), ("A_exp", st["A_exposed"]),
                         ("ps", st["ps"])):
                rec[k].append(v)
            if st["qL"] < self.opts.q_stop:
                h_stop = h
                break
            last_ok = (h, breached.copy())
        if h_stop is None:
            h_stop = 0.0
        elif refine and last_ok is not None:
            lo, hi, b_ok = h_stop, last_ok[0], last_ok[1]
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                st = self.state(mid + self.Z_min, b_ok.copy())
                if st["qL"] < self.opts.q_stop:
                    lo = mid
                else:
                    hi, b_ok = mid, st["breached"]
            h_stop = 0.5 * (lo + hi)
        out = {k: np.array(v, float) for k, v in rec.items()}
        out["h_stop"] = float(h_stop)
        out["q0"] = float(out["qL"][0])
        out["t"] = self.drain_time(out["h"], out["qL"])
        return out

    # -------------------------------------------------------------- pool geometry and time
    def free_surface_area(self, h):
        """Pool free-surface area (m2) at level h above the drain's lowest point."""
        H = h + self.Z_min
        x = np.linspace(0, self.drain.length, 241)
        zl = (H - x * self.sa) / self.ca                  # level in the drain's local z
        c = np.clip((zl - self.drain.b) / self.drain.b, -1, 1)
        chord = np.where((zl > 0) & (zl < 2 * self.drain.b), 2 * self.drain.a * np.sqrt(1 - c * c), 0.0)
        foot = np.trapezoid(chord, x) * 1e-6
        return self.opts.pool_area - foot

    def drain_time(self, h, qL):
        """Time (s) to reach each level, from dV = A_fs dh = -qL dt."""
        A = np.array([self.free_surface_area(v) for v in h])
        rate = np.where(qL > self.opts.q_stop, A / np.maximum(qL, 1e-30), np.nan) * 1e-3
        t = np.zeros_like(h)
        for i in range(1, len(h)):
            r0, r1 = rate[i - 1], rate[i]
            if not (np.isfinite(r0) and np.isfinite(r1)):
                t[i:] = np.nan
                break
            t[i] = t[i - 1] + 0.5 * (r0 + r1) * (h[i - 1] - h[i])
        return t

    # -------------------------------------------------------------- closed-form criterion
    def analytic(self, dh=0.01, with_capillary=False):
        """Closed-form stop level. Drainage needs submerged area and enough lift:

            simple bound:  h + H_v / (1 + Lambda(h)^2)   > z_lip,  Lambda = sum(Cd A_exp) / (Cd_p A_p)
            refined:       h + H_v / (1 + Lambda_w(h)^2) > z_lip,  each exposed element weighted by
                           sqrt((min(z, z_lip) - h) / (z_lip - h)): air entering below the inner
                           liquid level has to push against that liquid column.
        """
        hs = np.arange(self.h_start(), -1e-9, -dh)
        Hv = self.Hv
        pc_floor = float(np.min(self.Pc)) / self.rgl if with_capillary else 0.0
        z_lip = self.Z_lip - self.Z_min
        Zr = self.Z - self.Z_min                                   # element heights above lowest point
        cA = (self.cd * self.A / self.opts.n_el)[:, None] / (self.opts.cd_port * self.Ap)
        thr = self.sc[:, None]
        lam, lamw, sub = [], [], []
        for h in hs:
            exp = (Zr >= h - thr) & ~self.blocked
            sm = (Zr < h - thr) & ~self.blocked
            lam.append(float(np.sum(cA * exp)))
            if h < z_lip:
                w = np.clip((np.minimum(Zr, z_lip) - h) / (z_lip - h), 0, 1)
                lamw.append(float(np.sum(cA * exp * np.sqrt(w))))
            else:
                lamw.append(lam[-1])
            sub.append(float(np.sum(sm * (self.A / self.opts.n_el)[:, None])))
        lam, lamw, sub = np.array(lam), np.array(lamw), np.array(sub)
        out = dict(h=hs, Lambda=lam, Lambda_w=lamw, A_sub=sub, z_lip=z_lip)
        for tag, L_ in (("", lamw), ("_simple", lam)):
            lift = np.maximum(Hv / (1 + L_ ** 2), pc_floor)
            flowing = (sub > 0) & (hs + lift > z_lip)
            idx = np.where(~flowing)[0]
            out["lift" + tag] = lift
            out["h_stop" + tag] = float(hs[idx[0]]) if idx.size else 0.0
        sub_idx = np.where(sub > 0)[0]
        out["h_geo"] = float(hs[sub_idx[-1]]) if sub_idx.size else float(hs[0])
        return out

    # -------------------------------------------------------------- handy numbers
    def q_port_max(self):
        return self.opts.cd_port * self.Ap * math.sqrt(-2 * self.suction.p_set / self.fluid.rho)

    def lip_height(self):
        return self.Z_lip - self.Z_min

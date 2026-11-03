"""Time-dependent simulation of the drain emptying a pool (second, independent formulation).

Unlike DrainModel.descend(), which assumes that at each pool level the drain interior is in
steady state (liquid in = liquid out, gas in = gas out), this model keeps the liquid stored
inside the drain as a state variable:

    dV_L/dt = q_L,in - q_leak - q_L,bore            (liquid inside the drain)
    dH/dt   = -(q_L,in - q_leak) / A_pool(H)       (pool level)
    0       = q_L,in + q_G,in - q_leak - q_L,bore - q_G,bore    (rigid drain, volume conserved)

The last line fixes the gas pressure p_g at every instant (gas compressibility is negligible
at 1 % pressure changes and acts on micro-second scales). The inner liquid level zeta follows
from V_L through the interior geometry. V_L is advanced with backward Euler (stable for the
stiff bore), H explicitly with a level-change limited time step. Openings break through to air
when the suction at their inner side exceeds their capillary entry pressure, and close again
when they are submerged.
"""
import math
import numpy as np

from .model import illinois, segment_fraction


def interior_table(m, n=1500):
    """Liquid volume inside the drain (m3) versus a horizontal inner level zeta (world mm)."""
    dr = m.drain
    ai, bi = dr.a - dr.wall, dr.b - dr.wall
    x = np.linspace(0, dr.length, 301)
    z_bot = min(x[0] * m.sa, x[-1] * m.sa) + dr.wall * m.ca
    z_top = max(x[0] * m.sa, x[-1] * m.sa) + (2 * dr.b - dr.wall) * m.ca
    zeta = np.linspace(z_bot, z_top, n)
    zl = (zeta[:, None] - x[None, :] * m.sa) / m.ca                 # local level along the drain
    area = math.pi * ai * bi * segment_fraction((zl - dr.b) / bi)    # mm2
    V = np.trapezoid(area, x, axis=1) * 1e-9                         # m3
    return zeta, V


def simulate(m, t_end=1800.0, dh_max=0.05, record_every=1.0):
    fs_e = m.exposure
    zeta_tab, V_tab = interior_table(m)
    V_int = V_tab[-1]
    z_bot, z_top = zeta_tab[0], zeta_tab[-1]
    ps = m.suction.p_set
    rgl = m.rgl

    def V_of(z):
        return float(np.interp(z, zeta_tab, V_tab))

    def flows(H, ex, open_, zeta, pg):
        fs, fe, zs, ze = ex
        p_sub = pg + rgl * np.maximum(zeta - zs, 0.0)
        p_exp = pg + rgl * np.maximum(zeta - ze, 0.0)
        qLin = m._liquid_in(H, fs, zs, p_sub).sum()
        qGin = m._gas_in(fe, open_, p_exp).sum()
        qleak = m._leak_out(fe, p_exp, ze < zeta).sum()
        qLp, qGp = m._port(pg, zeta, ps)
        return qLin, qGin, qleak, qLp, qGp

    def gas_pressure(H, ex, open_, zeta):
        def f(pg):
            a, b, c, d, e = flows(H, ex, open_, zeta, pg)
            return a + b - c - d - e
        lo, hi = ps, rgl * (H - m.Z_min) + 50.0
        flo, fhi = f(lo), f(hi)
        if flo <= 0:
            return lo
        if fhi >= 0:
            return hi
        return illinois(f, lo, hi, flo, fhi, 1e-6)

    def net_liquid(H, ex, open_, zeta):
        pg = gas_pressure(H, ex, open_, zeta)
        qLin, qGin, qleak, qLp, qGp = flows(H, ex, open_, zeta, pg)
        return qLin - qleak - qLp, pg, (qLin, qGin, qleak, qLp, qGp)

    H = m.h_start() + m.Z_min
    zeta = z_top
    V = V_int
    open_ = np.zeros(m.pattern.n, bool)
    t, dt = 0.0, 0.05
    tr = {k: [] for k in ("t", "h", "zeta", "pg", "qL_in", "qG_in", "qL_out", "qG_out")}
    next_rec = 0.0
    stalled_since = None

    while t < t_end:
        ex = fs_e(H)
        fs, fe = ex[0], ex[1]
        open_ = open_ & (fe > 0)                         # submerged openings close again
        # breakthrough check at the current state
        pg = gas_pressure(H, ex, open_, zeta)
        p_exp = pg + rgl * np.maximum(zeta - ex[3], 0.0)
        new = (fe > 0) & ~open_ & (-p_exp > m.Pc)
        if np.any(new):
            open_ = open_ | new
            continue

        # implicit step for the stored liquid: V(zeta1) - V - dt * N(zeta1) = 0
        def R(z1):
            return V_of(z1) - V - dt * net_liquid(H, ex, open_, z1)[0]
        r_lo, r_hi = R(z_bot), R(z_top)
        if r_hi <= 0:
            z1 = z_top
        elif r_lo >= 0:
            z1 = z_bot
        else:
            z1 = illinois(R, z_bot, z_top, r_lo, r_hi, 1e-6)
        N, pg, q = net_liquid(H, ex, open_, z1)
        qLin, qGin, qleak, qLp, qGp = q
        A_fs = m.free_surface_area(H - m.Z_min)
        dHdt = -(qLin - qleak) / A_fs * 1e3              # mm/s

        if t >= next_rec:
            for k, v in (("t", t), ("h", H - m.Z_min), ("zeta", z1 - m.Z_min), ("pg", pg),
                         ("qL_in", qLin - qleak), ("qG_in", qGin), ("qL_out", qLp), ("qG_out", qGp)):
                tr[k].append(v)
            next_rec = t + record_every

        zeta, V = z1, V_of(z1)
        H += dHdt * dt
        t += dt
        if H < m.Z_min:
            H = m.Z_min
            break
        # step size: limit the pool-level change, grow when quiet
        rate = abs(dHdt)
        dt = min(5.0, dh_max / rate if rate > 0 else 5.0, dt * 1.5)
        dt = max(dt, 1e-3)
        if qLp <= m.opts.q_stop and abs(qLin - qleak) <= m.opts.q_stop:
            stalled_since = t if stalled_since is None else stalled_since
            if t - stalled_since > 120.0:
                break
        else:
            stalled_since = None

    out = {k: np.array(v, float) for k, v in tr.items()}
    return dict(trace=out, h_final=float(H - m.Z_min), t_final=t,
                stalled=stalled_since is not None)

"""
Reduced-order (axisymmetric) version of the coupled problem, solved monolithically in
Python. Use it to choose parameters, to sanity-check the OpenFOAM/FEBio runs, and for
the pictures. It is NOT a replacement for the 3D models (small-strain, round window,
flat skin, frictionless).

  tissue  r in [0, R], z in [0, H] above the drain: linear Biot poroelasticity
          (u-p, Taylor-Hood Q2/Q1), drained modulus E, Poisson nu, permeability k
  film    liquid layer h(r) = g0 + u_z(r, 0) between tissue and skin for r > a:
          lubrication (Reynolds) flow  Q' = -h^3/(12 mu) dp/dr, with squeeze term dh/dt
          and seepage from the tissue; p_film = pore pressure at the tissue surface.
          Solved as a 1D conduit on the tissue's bottom edge (same unknowns).
  hole    r < a: tissue surface sees the hole pressure  p_hole = -dp + R_hole * Q
          (Poiseuille through the wall + Sampson orifice)
  contact h >= h_min (roughness), penalty, frictionless
  edges   top: held, p = 0 (body);  r = R: u_r = 0, p = 0 (film edge open to liquid)
Units: mm, N, s, MPa.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geom import graded


# ------------------------------------------------------------------ shape functions
def _q2(xi, eta):
    """9-node Lagrange quad: N, dN/dxi, dN/deta. Node order: (i,j) for i,j in 0..2 (xi fastest)."""
    l = lambda s: np.array([0.5 * s * (s - 1), 1 - s * s, 0.5 * s * (s + 1)])
    dl = lambda s: np.array([s - 0.5, -2 * s, s + 0.5])
    Lx, Ly, dLx, dLy = l(xi), l(eta), dl(xi), dl(eta)
    N = np.outer(Ly, Lx).ravel()
    dNx = np.outer(Ly, dLx).ravel()
    dNy = np.outer(dLy, Lx).ravel()
    return N, dNx, dNy


def _q1(xi, eta):
    l = lambda s: np.array([0.5 * (1 - s), 0.5 * (1 + s)])
    dl = lambda s: np.array([-0.5, 0.5])
    Lx, Ly = l(xi), l(eta)
    return np.outer(Ly, Lx).ravel(), np.outer(Ly, dl(xi)).ravel(), np.outer(dl(eta), Lx).ravel()


GP3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
GW3 = np.array([5 / 9, 8 / 9, 5 / 9])


class Preview:
    def __init__(self, cfg, R=None, nr_hole=10, nr_out=34, nz=22):
        W = cfg["window"]
        self.cfg = cfg
        self.a = 0.5 * W.get("hole_d", 1.0)
        self.t_wall = W.get("wall", 0.6)
        self.fillet = W.get("fillet", 0.0)
        half = W.get("half", 1.6)
        self.R = R or 2 * half / np.sqrt(np.pi)                  # same plan area as the square window
        self.H = W.get("tissue_h", 2.0)
        # standoff ring (new offset design): ring top at hs, tissue starts gap above it
        self.hs = W.get("standoff_h", 0.0)
        self.rin, self.rout = W.get("standoff_rin", 0.55), W.get("standoff_rout", 0.85)
        self.slot_w, self.n_slots = W.get("slot_w", 0.35), W.get("n_slots", 4)
        self.gap = W.get("gap", 0.05)
        self.g0 = self.gap + self.hs                              # initial tissue height above the skin
        self.hmin = W.get("h_min", 0.003)
        T = cfg["tissue"]
        self.E = T["E_Pa"] * 1e-6
        self.nu = T["nu"]
        self.k = T["perm_m4_Ns"] * 1e12
        self.mu = cfg["fluid"]["mu_Pa_s"] * 1e-6
        self.dp = cfg["suction"]["dp_Pa"] * 1e-6
        self.ramp = cfg["suction"].get("ramp_s", 0.01)
        a = self.a
        self.R_hole = 8 * self.mu * self.t_wall / (np.pi * a ** 4) + 3 * self.mu / a ** 3
        # inertial (orifice) loss of the hole: dp = rho Q^2 / (2 Cd^2 A^2), Cd = 0.61
        self.rho = cfg["fluid"]["rho_kg_m3"] * 1e-12                 # N s^2 / mm^4
        self.K_hole = self.rho / (2 * 0.61 ** 2 * (np.pi * a * a) ** 2)
        # free liquid at the window edge: "open" = pool at body pressure, "closed" = none
        self.film_edge_open = cfg.get("boundary", {}).get("film_edge", "open") == "open"
        # ---- element edges
        rh = a * (1 - graded(nr_hole, 0.03)[::-1])               # fine towards the rim
        if self.hs > 0:                                          # nodes exactly at the ring radii
            r_a = np.linspace(a, self.rin, 3)
            r_r = np.linspace(self.rin, self.rout, 9)
            r_o = self.rout + (self.R - self.rout) * graded(nr_out - 10, 0.004 / (self.R - self.rout) * 2)
            ro = np.concatenate([r_a, r_r[1:], r_o[1:]])
        else:
            ro = a + (self.R - a) * graded(nr_out, 0.004 / (self.R - a) * 2)
        self.re = np.concatenate([rh, ro[1:]])
        self.ze = self.H * graded(nz, 0.012 / self.H * 2)
        self._mesh()
        self._assemble_static()

    # -------------------------------------------------------------- mesh
    def _mesh(self):
        re, ze = self.re, self.ze
        nr, nz = len(re) - 1, len(ze) - 1
        # Q2 node grid
        r2 = np.empty(2 * nr + 1); r2[::2] = re; r2[1::2] = 0.5 * (re[1:] + re[:-1])
        z2 = np.empty(2 * nz + 1); z2[::2] = ze; z2[1::2] = 0.5 * (ze[1:] + ze[:-1])
        self.r2, self.z2 = r2, z2
        self.n2r, self.n2z = len(r2), len(z2)
        self.n2 = self.n2r * self.n2z
        self.n1r, self.n1z = nr + 1, nz + 1
        self.n1 = self.n1r * self.n1z
        id2 = lambda i, j: j * self.n2r + i
        id1 = lambda i, j: j * self.n1r + i
        E2, E1 = [], []
        for ej in range(nz):
            for ei in range(nr):
                E2.append([id2(2 * ei + di, 2 * ej + dj) for dj in range(3) for di in range(3)])
                E1.append([id1(ei + di, ej + dj) for dj in range(2) for di in range(2)])
        self.E2, self.E1 = np.array(E2), np.array(E1)
        self.nu_dof = 2 * self.n2
        self.ndof = self.nu_dof + self.n1
        R2, Z2 = np.meshgrid(r2, z2)
        self.X2 = np.c_[R2.ravel(), Z2.ravel()]
        R1, Z1 = np.meshgrid(re, ze)
        self.X1 = np.c_[R1.ravel(), Z1.ravel()]
        # bottom nodes
        self.bot2 = np.arange(self.n2r)                          # z = 0 row of Q2 nodes
        self.bot1 = np.arange(self.n1r)
        self.zb = np.zeros(self.n2r)                             # skin height under each bottom node
        f = self.fillet
        if f > 0:
            d = np.clip(self.a + f - r2, 0, f)
            self.zb = -(f - np.sqrt(np.clip(f * f - d * d, 0, None)))
            self.zb[r2 < self.a] = 0.0
        # standoff: ring top at hs; phi = share of the circumference that is slot (open)
        self.ring2 = (r2 >= self.rin - 1e-12) & (r2 <= self.rout + 1e-12) if self.hs > 0 else np.zeros(len(r2), bool)
        self.zb = np.where(self.ring2, self.hs, self.zb)
        rm = 0.5 * (re[1:] + re[:-1])
        self.ring_e = (rm > self.rin) & (rm < self.rout) if self.hs > 0 else np.zeros(nr, bool)
        self.phi_e = np.clip(self.n_slots * self.slot_w / (2 * np.pi * rm), 0, 1) * self.ring_e
        self.support2 = np.ones(len(r2))                         # share of the area that can carry contact
        self.support2[self.ring2] = 1 - np.clip(self.n_slots * self.slot_w / (2 * np.pi * r2[self.ring2]), 0, 1)

    # -------------------------------------------------------------- static matrices
    def _assemble_static(self):
        E, nu, k = self.E, self.nu, self.k
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        G = E / (2 * (1 + nu))
        D = np.array([[lam + 2 * G, lam, lam, 0], [lam, lam + 2 * G, lam, 0],
                      [lam, lam, lam + 2 * G, 0], [0, 0, 0, G]])
        rows, cols, vK = [], [], []
        rC, cC, vC = [], [], []
        rB, cB, vB = [], [], []
        rH, cH, vH = [], [], []
        for e2, e1 in zip(self.E2, self.E1):
            x2 = self.X2[e2]
            Ke = np.zeros((18, 18)); Ce = np.zeros((18, 4)); Be = np.zeros((4, 18)); He = np.zeros((4, 4))
            for gi, wi in zip(GP3, GW3):
                for gj, wj in zip(GP3, GW3):
                    N, dNx, dNy = _q2(gi, gj)
                    J = np.array([[dNx @ x2[:, 0], dNx @ x2[:, 1]], [dNy @ x2[:, 0], dNy @ x2[:, 1]]])
                    dJ = np.linalg.det(J)
                    Ji = np.linalg.inv(J)
                    dN = Ji @ np.vstack([dNx, dNy])               # d/dr, d/dz
                    r = N @ x2[:, 0]
                    w = wi * wj * dJ * r
                    B = np.zeros((4, 18))
                    B[0, 0::2] = dN[0]; B[1, 1::2] = dN[1]; B[2, 0::2] = N / r
                    B[3, 0::2] = dN[1]; B[3, 1::2] = dN[0]
                    Ke += w * B.T @ D @ B
                    M, dMx, dMy = _q1(gi, gj)
                    dM = Ji @ np.vstack([dMx, dMy])
                    gradu = np.zeros((18, 4))                     # v . grad p
                    gradu[0::2] = np.outer(N, dM[0]); gradu[1::2] = np.outer(N, dM[1])
                    Ce += w * gradu
                    div = B[0] + B[1] + B[2]
                    Be += w * np.outer(M, div)
                    He += w * k * dM.T @ dM
            dofs = np.empty(18, int); dofs[0::2] = 2 * e2; dofs[1::2] = 2 * e2 + 1
            pd = self.nu_dof + e1
            rows += np.repeat(dofs, 18).tolist(); cols += np.tile(dofs, 18).tolist(); vK += Ke.ravel().tolist()
            rC += np.repeat(dofs, 4).tolist(); cC += np.tile(pd, 18).tolist(); vC += Ce.ravel().tolist()
            rB += np.repeat(pd, 18).tolist(); cB += np.tile(dofs, 4).tolist(); vB += Be.ravel().tolist()
            rH += np.repeat(pd, 4).tolist(); cH += np.tile(pd, 4).tolist(); vH += He.ravel().tolist()
        n = self.ndof
        self.K = sp.csr_matrix((vK + vC, (rows + rC, cols + cC)), shape=(n, n))
        self.B = sp.csr_matrix((vB, (rB, cB)), shape=(n, n))      # div u rows
        self.Hk = sp.csr_matrix((vH, (rH, cH)), shape=(n, n))
        # bottom-edge 1D integrals: lumped area per Q2 bottom node, squeeze (q * u_z) and film conduction
        r2 = self.r2
        nr = len(self.re) - 1
        self.area2 = np.zeros(self.n2r)
        rS, cS, vS = [], [], []
        for ei in range(nr):
            i2 = [2 * ei, 2 * ei + 1, 2 * ei + 2]
            i1 = [ei, ei + 1]
            xr = r2[i2]
            for g, wg in zip(GP3, GW3):
                L2 = np.array([0.5 * g * (g - 1), 1 - g * g, 0.5 * g * (g + 1)])
                L1 = np.array([0.5 * (1 - g), 0.5 * (1 + g)])
                dr = np.array([g - 0.5, -2 * g, g + 0.5]) @ xr
                r = L2 @ xr
                w = wg * dr * r
                self.area2[i2] += w * L2
                for a_, la in zip(i1, L1):
                    for b_, lb in zip(i2, L2):
                        rS.append(self.nu_dof + a_); cS.append(2 * b_ + 1); vS.append(w * la * lb)
        self.S = sp.csr_matrix((vS, (rS, cS)), shape=(n, n))       # int q u_z r dr (bottom)

    def film_conductance(self, h_mid):
        """Radial film conductance per unit circumference, per element (mm^5/(N s)).
        Ring elements: (1-phi) of the circumference is ring top (film h above it), phi is slot:
        a rectangular duct of width slot_w and height hs + h (exact series, 1st term)."""
        c = h_mid ** 3 / (12 * self.mu)
        if self.hs > 0 and np.any(self.ring_e):
            H = self.hs + h_mid
            b, d = np.minimum(H, self.slot_w), np.maximum(H, self.slot_w)
            q_duct = d * b ** 3 / (12 * self.mu) * (1 - 0.630 * b / d)       # per unit pressure gradient
            c_slot = q_duct / self.slot_w                                  # per unit slot width
            ph = self.phi_e
            c = np.where(self.ring_e, (1 - ph) * c + ph * c_slot, c)
        return c

    def _film_matrix(self, cbar):
        """Delta-t free film conduction matrix: int c(h) dq/dr dp/dr r dr on r > a (per element, c at midpoint)."""
        re = self.re
        rows, cols, vals = [], [], []
        for ei in range(len(re) - 1):
            r0, r1 = re[ei], re[ei + 1]
            if r0 < self.a - 1e-12:
                continue
            c = cbar[ei]
            kk = c * 0.5 * (r0 + r1) / (r1 - r0)
            i, j = self.nu_dof + ei, self.nu_dof + ei + 1
            rows += [i, i, j, j]; cols += [i, j, i, j]; vals += [kk, -kk, -kk, kk]
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof))

    # -------------------------------------------------------------- solve in time
    def run(self, t_end=3600.0, n_steps=140, t_first=1e-5, snap_times=None, verbose=True, side_drained=None):
        if side_drained is None:
            side_drained = self.film_edge_open
        a = self.a
        n = self.ndof
        nud = self.nu_dof
        X2, X1 = self.X2, self.X1
        # Dirichlet sets
        top2 = np.nonzero(np.isclose(X2[:, 1], self.H))[0]
        axis2 = np.nonzero(X2[:, 0] < 1e-12)[0]
        side2 = np.nonzero(np.isclose(X2[:, 0], self.R))[0]
        fix_u = np.unique(np.concatenate([2 * top2, 2 * top2 + 1, 2 * axis2, 2 * side2]))
        top1 = np.nonzero(np.isclose(X1[:, 1], self.H))[0]
        side1 = np.nonzero(np.isclose(X1[:, 0], self.R))[0]
        edge = side1 if side_drained else (side1[:1] if self.film_edge_open else side1[:0])
        zero_p = nud + np.unique(np.concatenate([top1, edge]))
        hole1 = np.nonzero(np.isclose(X1[:, 1], 0) & (X1[:, 0] <= a + 1e-12))[0]
        hole_p = nud + hole1
        bot_out2 = np.nonzero(self.r2 >= a - 1e-12)[0]            # contact candidates (Q2 bottom row)
        free = np.ones(n, bool); free[fix_u] = False; free[zero_p] = False; free[hole_p] = False
        fi = np.nonzero(free)[0]
        times = np.concatenate([[0.0], t_first * (t_end / t_first) ** (np.arange(n_steps) / (n_steps - 1))])
        x = np.zeros(n)
        kpen = 50 * self.E / (self.ze[1] - self.ze[0])
        hist = dict(t=[], p_hole=[], Q_hole=[], Q_edge=[], Q_seep=[], Q_body=[], mass_err=[], V_drained=[], h_rim=[], contact=[], clearance=[], ring_load=[],
                    intrusion=[], contact_r=[], sealed=[], contact_force=[])
        snaps = []
        snap_times = sorted(snap_times or [])
        p_hole = 0.0
        ubot = lambda xx: xx[2 * self.bot2 + 1]
        active = np.zeros(self.n2r, bool)
        V0 = 0.0
        for s in range(1, len(times)):
            t0, t1 = times[s - 1], times[s]
            dt = t1 - t0
            ramp = min(1.0, t1 / self.ramp)
            xn = x.copy()
            h_lag = np.maximum(self.g0 + ubot(xn) - self.zb, self.hmin)
            for it in range(60):
                h_mid = 0.5 * (h_lag[0:-2:2] + h_lag[2::2])
                cfilm = self.film_conductance(h_mid)
                F = self._film_matrix(cfilm)
                A = self.K + self.B + dt * self.Hk + self.S + dt * F
                # contact penalty
                pen = np.zeros(n)
                dpen = np.zeros(n)
                idx = 2 * self.bot2 + 1
                ka = kpen * self.area2 * self.support2
                dpen[idx[active]] = ka[active]
                pen[idx[active]] = ka[active] * (self.hmin - self.g0 + self.zb[active])
                A = A + sp.diags(dpen)
                b = (self.B + self.S) @ xn + pen
                # the state is affine in the hole pressure: x = x0 + p_hole * x1
                lu = spla.splu(A[fi][:, fi].tocsc())
                x0 = np.zeros(n)
                x0[fi] = lu.solve(b[fi])
                e = np.zeros(n); e[hole_p] = 1.0
                x1 = e.copy()
                x1[fi] = lu.solve(-(A @ e)[fi])
                Qf = lambda xx: -2 * np.pi * (A @ xx - b)[hole_p].sum() / dt      # into the hole, mm^3/s
                Q0 = Qf(x0)
                G = -2 * np.pi * (A @ x1)[hole_p].sum() / dt
                # hole law p = p_ch + R Q + K Q|Q| with Q = Q0 + G p: monotone in p -> bisection
                p_ch = ramp * (-self.dp)
                f = lambda p: p - p_ch - self.R_hole * (Q0 + G * p) - self.K_hole * (Q0 + G * p) * abs(Q0 + G * p)
                lo, hi = min(p_ch, 0.0) - 1e-12, max(p_ch, 0.0) + 1e-12
                while f(lo) > 0:
                    lo -= abs(p_ch) + 1e-9
                while f(hi) < 0:
                    hi += abs(p_ch) + 1e-9
                for _ in range(100):
                    mid = 0.5 * (lo + hi)
                    if f(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                p_new = 0.5 * (lo + hi)
                xs = x0 + p_new * x1
                Q_hole = Q0 + G * p_new
                h_new = self.g0 + ubot(xs) - self.zb
                act_new = np.zeros(self.n2r, bool)
                act_new[bot_out2] = h_new[bot_out2] < self.hmin
                act_new |= active & (h_new < self.hmin * 1.5)      # hysteresis
                dh = np.max(np.abs(np.maximum(h_new, self.hmin) - h_lag))
                conv = (dh < 1e-4 * self.g0 + 1e-7) and np.array_equal(act_new, active) \
                    and abs(p_new - p_hole) < 1e-4 * self.dp
                h_lag = 0.5 * h_lag + 0.5 * np.maximum(h_new, self.hmin) if it > 8 else np.maximum(h_new, self.hmin)
                active = act_new
                p_hole = p_new
                x = xs
                if conv and it > 0:
                    break
            res = A @ x - b
            corner = nud + side1[0]                                  # film edge (bottom corner)
            Q_edge = 2 * np.pi * res[corner] / dt if self.film_edge_open else 0.0   # free liquid entering the film
            body = zero_p[zero_p != corner]
            Q_body = 2 * np.pi * res[body].sum() / dt                # body liquid entering through top/sides
            dV = -2 * np.pi * np.sum(self.B[nud:].dot(x - xn))       # liquid squeezed out of the sponge (mm^3)
            V0 += dV
            hn_ = np.maximum(self.g0 + ubot(xn) - self.zb, 0)
            h1_ = np.maximum(self.g0 + ubot(x) - self.zb, 0)
            o = self.r2 >= a - 1e-12
            dVfilm = 2 * np.pi * np.sum(self.area2[o] * (h1_ - hn_)[o])
            dVhole = 2 * np.pi * np.sum(self.area2[~o] * (ubot(x) - ubot(xn))[~o])
            hist["Q_body"].append(Q_body)
            hist["mass_err"].append((Q_hole - (Q_edge + Q_body + dV / dt - dVfilm / dt - dVhole / dt))
                                    / max(abs(Q_hole), 1e-30))
            ub = ubot(x)
            h = self.g0 + ub - self.zb
            rim_i = np.argmin(np.abs(self.r2 - a))
            intr = float(-ub[self.r2 < a].min() - self.g0) if np.any(self.r2 < a) else 0.0
            cont = np.nonzero(active)[0]
            hist["t"].append(t1); hist["p_hole"].append(p_hole * 1e6); hist["Q_hole"].append(Q_hole)
            hist["Q_edge"].append(Q_edge)
            hist["Q_seep"].append(dV / dt); hist["V_drained"].append(V0)
            hist["h_rim"].append(float(h[rim_i])); hist["intrusion"].append(intr)
            hist["contact_r"].append([float(self.r2[cont].min()), float(self.r2[cont].max())] if len(cont) else None)
            ring = active[(self.r2 >= a) & (self.r2 <= a + 0.25)]
            hist["sealed"].append(bool(len(cont) and ring[0]) and self.hs == 0)   # slots never seal
            hist["contact"].append(bool(len(cont)))
            inside = self.r2 < (self.rin if self.hs > 0 else a)
            hist["clearance"].append(float((self.g0 + ub[inside]).min()))       # lowest tissue point over the cup/hole vs skin level
            hist["ring_load"].append(float(2 * np.pi * np.sum((kpen * self.area2 * self.support2 *
                                     np.maximum(self.hmin - h, 0))[self.ring2])) if self.hs > 0 else 0.0)
            hist["contact_force"].append(float(2 * np.pi * np.sum(kpen * self.area2[active] * self.support2[active] *
                                               (self.hmin - h[active]))))
            while snap_times and t1 >= snap_times[0] * 0.999:
                snaps.append(self.fields(x, t1, p_hole))
                snap_times.pop(0)
            if verbose and (s % 10 == 0 or s == len(times) - 1):
                print(f"t={t1:9.3g}s it={it:2d} p_hole={p_hole*1e3:7.3f}kPa Q_hole={Q_hole*0.06:9.3g}mL/min "
                      f"h_rim={h[rim_i]*1e3:6.2f}um intr={intr*1e3:6.1f}um V_drained={V0:8.4f}mm3 "
                      f"contact={hist['contact_r'][-1]}")
        self.x = x
        return hist, snaps

    # -------------------------------------------------------------- fields for plotting
    def fields(self, x, t, p_hole):
        nud = self.nu_dof
        u = x[:nud].reshape(-1, 2)
        p1 = x[nud:]
        # pore pressure on the Q2 grid (bilinear from Q1)
        from scipy.interpolate import RegularGridInterpolator
        P = p1.reshape(self.n1z, self.n1r)
        ip = RegularGridInterpolator((self.ze, self.re), P)
        p2 = ip(self.X2[:, ::-1])
        # Darcy flux at element centres
        wc, xc = [], []
        for e2, e1 in zip(self.E2, self.E1):
            x2 = self.X2[e2]
            N, dNx, dNy = _q2(0, 0)
            J = np.array([[dNx @ x2[:, 0], dNx @ x2[:, 1]], [dNy @ x2[:, 0], dNy @ x2[:, 1]]])
            M, dMx, dMy = _q1(0, 0)
            dM = np.linalg.inv(J) @ np.vstack([dMx, dMy])
            wc.append(-self.k * dM @ p1[e1])
            xc.append(N @ x2)
        ub = u[self.bot2]
        h = self.g0 + ub[:, 1] - self.zb
        return dict(t=float(t), p_hole_Pa=float(p_hole * 1e6), r2=self.r2, z2=self.z2,
                    u=u.reshape(self.n2z, self.n2r, 2), p_Pa=p2.reshape(self.n2z, self.n2r) * 1e6,
                    w_mm_s=np.array(wc), xc=np.array(xc), film_h=np.maximum(h, 0),
                    film_p_Pa=p2.reshape(self.n2z, self.n2r)[0] * 1e6)

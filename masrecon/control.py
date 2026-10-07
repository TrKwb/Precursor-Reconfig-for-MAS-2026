# -*- coding: utf-8 -*-
"""Task C : distributed MPC with discrete-time CBF constraints (1-D, exact linear).

Decision variable z = [U (Nh inputs), xi (ONE scalar slack)].  The slack is a
single non-negative scalar per agent and per instant, shared by every softened
row (all stages, all neighbours) and penalised by slack_w * xi^2; it is the
L-infinity relaxation of (9), not a per-stage vector.

Options in cfg (all default OFF -> legacy behaviour, bit-identical):
  hard_barrier    two-tier constraint (Sec. 5.6(b)): h(t+1) >= 0 hard, decay soft.
                  If the hard problem is infeasible the soft one is solved and
                  hard_relaxed is incremented.
  nonretreat      Assumption 2 imposed in the agent's OWN problem:
                  v_i(t) >= 0, t = 1..Nh (hard rows).  Together with the input
                  model this implies p_i(t+1) >= p_i(t), so no position rows
                  are needed.  If the problem with these rows is infeasible
                  (cannot happen from v_i(0) >= 0, since maximal admissible
                  braking satisfies them), the rows are dropped and
                  nonretreat_relaxed is incremented.
  terminal_kappa  float kappa.  For every FORWARD neighbour whose tube is
                  constant over the horizon (stop tube), add the terminal row
                  hbar_i(Nh) >= kappa * v_i(Nh).  Omega_stop = {h >= kappa v} is
                  invariant under maximal braking for kappa >= 0.5 (verified
                  offline), but with Nh = 10 it is too small to be reachable
                  from states that are stop-feasible to rest: NOT recommended on
                  the benchmark; the runner's shield enforces the exact set.
                  Softened with the same slack unless hard_barrier is set.

Diagnostics of the LAST call: self.last = dict(ok, slack, hard_ok, n_rows).
"""
import numpy as np
from .qp import solve_qp


class Controller:
    def __init__(self, cfg):
        self.c = cfg
        Nh, dt = cfg.ctrl.Nh, cfg.plant.dt
        cdr = cfg.plant.drag
        A = np.array([[1.0, dt], [0.0, 1.0 - cdr * dt]])
        B = np.array([[0.5 * dt * dt], [dt]])
        self.Php = np.zeros((Nh, 2)); self.Phv = np.zeros((Nh, 2))
        self.Gp = np.zeros((Nh, Nh)); self.Gv = np.zeros((Nh, Nh))
        Ak = np.eye(2)
        for t in range(Nh):
            Ak = Ak @ A
            self.Php[t] = Ak[0]; self.Phv[t] = Ak[1]
            for s in range(t + 1):
                M = np.linalg.matrix_power(A, t - s) @ B
                self.Gp[t, s] = M[0, 0]; self.Gv[t, s] = M[1, 0]
        cc = cfg.ctrl
        self.H0 = 2.0 * (cc.q_p * self.Gp.T @ self.Gp
                         + cc.q_v * self.Gv.T @ self.Gv + cc.r_u * np.eye(Nh))
        self.Nh = Nh
        self.dp = np.zeros(Nh); self.dv = np.zeros(Nh)
        self.hard_relaxed = 0
        self.nonretreat_relaxed = 0
        self.last = {}

    def ballistic(self, x):
        return np.concatenate([[x[0]], self.Php @ x])

    def ballistic_v(self, x):
        return np.concatenate([[x[1]], self.Phv @ x])

    def _cbf(self, xi, pj, sign):
        """sign=+1 : neighbour j is AHEAD of i ; sign=-1 : j is BEHIND i."""
        Nh, g = self.Nh, self.c.ctrl.gamma
        dmin = self.c.plant.d_min
        rows = np.zeros((Nh, Nh)); rhs = np.zeros(Nh)
        for t in range(Nh):
            gp_n = self.Gp[t]
            gp_c = self.Gp[t - 1] if t >= 1 else np.zeros(Nh)
            ph_n = self.Php[t] @ xi + self.dp[t]
            ph_c = (self.Php[t - 1] @ xi + self.dp[t - 1]) if t >= 1 else xi[0]
            rows[t] = sign * (gp_n - (1 - g) * gp_c)
            rhs[t] = (sign * (pj[t + 1] - (1 - g) * pj[t]) - g * dmin
                      - sign * (ph_n - (1 - g) * ph_c))
        return rows, rhs

    def _hard(self, xi, pj, sign):
        """Hard barrier  sign*(p_j - p_i)(t+1) >= d_min  (two-tier, Sec. 5.6(b))."""
        Nh = self.Nh
        dmin = self.c.plant.d_min
        rows = np.zeros((Nh, Nh)); rhs = np.zeros(Nh)
        for t in range(Nh):
            ph_n = self.Php[t] @ xi + self.dp[t]
            rows[t] = sign * self.Gp[t]
            rhs[t] = sign * pj[t + 1] - sign * ph_n - dmin
        return rows, rhs

    def _nonretreat(self, xi):
        """v_i(t) >= 0, t = 1..Nh   ->   -Gv U <= Phv xi + dv."""
        return -self.Gv.copy(), self.Phv @ xi + self.dv

    def _terminal(self, xi, pj, kappa):
        """pj(Nh) - p_i(Nh) - d_min >= kappa v_i(Nh)  (forward, stop tube)."""
        dmin = self.c.plant.d_min
        row = self.Gp[-1] + kappa * self.Gv[-1]
        rhs = (pj[-1] - dmin - (self.Php[-1] @ xi + self.dp[-1])
               - kappa * (self.Phv[-1] @ xi + self.dv[-1]))
        return row[None, :], np.array([rhs])

    def solve(self, xi, pref, vref, nb):
        """nb : dict  j -> (predicted position trajectory length Nh+1, sign)"""
        Nh = self.Nh; cc = self.c.ctrl
        ep = self.Php @ xi + self.dp - pref
        ev = self.Phv @ xi + self.dv - vref
        f0 = 2.0 * (cc.q_p * self.Gp.T @ ep + cc.q_v * self.Gv.T @ ev)
        n = Nh + 1
        H = np.zeros((n, n)); H[:Nh, :Nh] = self.H0; H[Nh, Nh] = 2.0 * cc.slack_w
        f = np.zeros(n); f[:Nh] = f0
        umax = self.c.plant.u_max
        A = [np.hstack([np.eye(Nh), np.zeros((Nh, 1))]),
             np.hstack([-np.eye(Nh), np.zeros((Nh, 1))]),
             np.hstack([np.zeros((1, Nh)), -np.ones((1, 1))])]
        b = [umax * np.ones(Nh), umax * np.ones(Nh), np.zeros(1)]
        Ah, bh = [], []
        hard = bool(getattr(self.c, "hard_barrier", False))
#        nonret = bool(getattr(self.c, "nonretreat", False))
        nonret = bool(getattr(self.c, "nonretreat", False)) and bool(getattr(self, "nonret_active", True))
        kappa = getattr(self.c, "terminal_kappa", None)
        for j, (traj, sgn) in nb.items():
            R, r = self._cbf(xi, traj, sgn)
            A.append(np.hstack([R, -np.ones((Nh, 1))])); b.append(r)
            if hard:
                Hr, hb = self._hard(xi, traj, sgn)
                Ah.append(np.hstack([Hr, np.zeros((Nh, 1))])); bh.append(hb)
            if kappa is not None and sgn > 0 and np.ptp(traj) == 0.0:
                Tr, tb = self._terminal(xi, traj, float(kappa))
                if hard:
                    Ah.append(np.hstack([Tr, np.zeros((1, 1))])); bh.append(tb)
                else:
                    A.append(np.hstack([Tr, -np.ones((1, 1))])); b.append(tb)

        # Assumption 2 rows (hard, both tiers)
        An, bn = [], []
        if nonret:
            Nr, nr = self._nonretreat(xi)
            An.append(np.hstack([Nr, np.zeros((Nh, 1))])); bn.append(nr)

        def _solve(rows, rhs):
            return solve_qp(H, f, np.vstack(rows), np.concatenate(rhs))

        hard_ok = None
        if hard and Ah:
            z, ok = _solve(A + Ah + An, b + bh + bn)
            hard_ok = ok
            if not ok:                       # hard version infeasible -> soft
                self.hard_relaxed += 1
                z, ok = _solve(A + An, b + bn)
        else:
            z, ok = _solve(A + An, b + bn)
        if nonret and not ok:                # defensive: should not happen
            self.nonretreat_relaxed += 1
            z, ok = _solve(A, b)
        U = np.clip(z[:Nh], -umax, umax)
        self.last = dict(ok=bool(ok), slack=float(max(z[Nh], 0.0)), hard_ok=hard_ok,
                         n_rows=int(sum(len(x) for x in b + bh + bn)))
        return U, float(max(z[Nh], 0.0)), ok

    def predict(self, xi, U):
        return np.concatenate([[xi[0]], self.Php @ xi + self.Gp @ U + self.dp])
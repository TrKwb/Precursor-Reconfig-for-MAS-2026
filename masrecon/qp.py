# -*- coding: utf-8 -*-
"""Dense QP by primal active set.  min .5 x'Hx + f'x  s.t. Ax <= b,  H > 0.

Return value (x, ok).  ok is True only if x satisfies ALL rows of A x <= b to
within `feas_tol`.  (Legacy versions returned ok=True as soon as every row
outside the working set was satisfied, without re-checking the working set;
on an infeasible problem the regularised KKT solve can then return a point
that violates working-set rows, and the problem was reported as solved.)
The iterate x itself is unchanged with respect to the legacy version.

With return_info=True a third value is returned:
  dict(status, iters, maxviol)   status in {'optimal', 'infeasible_or_maxiter',
                                            'singular'}
"""
import numpy as np


def solve_qp(H, f, A, b, maxiter=50, tol=1e-8, feas_tol=1e-5, return_info=False):
    def out(x, status, it):
        mv = float(np.max(A @ x - b)) if (A is not None and len(b)) else 0.0
        ok = (status != "singular") and mv <= feas_tol
        if status == "optimal" and not ok:
            status = "infeasible_or_maxiter"
        return (x, ok, dict(status=status, iters=it, maxviol=mv)) if return_info else (x, ok)

    n = H.shape[0]
    try:
        x = np.linalg.solve(H, -f)
    except np.linalg.LinAlgError:
        return out(np.zeros(n), "singular", 0)
    if A is None or len(b) == 0:
        return (x, True, dict(status="optimal", iters=0, maxviol=0.0)) if return_info else (x, True)
    act = []
    it = 0
    for it in range(1, maxiter + 1):
        viol = A @ x - b
        cand = [i for i in range(len(b)) if i not in act]
        if not cand:
            break
        j = max(cand, key=lambda i: viol[i])
        if viol[j] <= tol:
            return out(x, "optimal", it)          # working set re-checked in out()
        act.append(j)
        for _ in range(maxiter):
            Aa = A[act]; m = len(act)
            K = np.zeros((n + m, n + m))
            K[:n, :n] = H; K[:n, n:] = Aa.T; K[n:, :n] = Aa
            K[n:, n:] = -1e-9 * np.eye(m)
            try:
                sol = np.linalg.solve(K, np.concatenate([-f, b[act]]))
            except np.linalg.LinAlgError:
                return out(x, "singular", it)
            xn, lam = sol[:n], sol[n:]
            if np.all(lam >= -tol):
                x = xn; break
            act.pop(int(np.argmin(lam)))
            if not act:
                x = np.linalg.solve(H, -f); break
    return out(x, "infeasible_or_maxiter", it)
# -*- coding: utf-8 -*-
"""Closed-loop runner.

The three methods differ ONLY in when the *belief* structure s_bel is updated:
  'fixed' : never; 'post' : at the instant of failure (oracle);
  'prop'  : when the conformal flag fires.
Physics, disturbances and diagnostics are identical (the plant is deep-copied),
so the comparison is paired.

Flags in cfg (all default OFF; used by exp9):
  force_critical   Remedy 2, handled in select.py
  fallback_tube    'stop' : when the critical edge to a forward neighbour is
                   absent, use the immediate-stop tube (Remark 3(i)) instead of
                   the invalid nominal-drift prediction (Remark 3(iii))
  always_stop      immediate-stop tube for EVERY forward neighbour (baseline)
  hard_barrier     two-tier constraint, handled in control.py
  freeze_structure tightening from detection, but structure changes only at
                   failure (no pre-emptive reconfiguration)
"""
import copy
import numpy as np
from .control import Controller
from .select import admissible_elements, select_structure
from .structure import neighbors


def _failing_actuator(plant):
    for nm in ("failing", "fail", "failures", "failing_elements"):
        v = getattr(plant, nm, None)
        if v:
            if isinstance(v, tuple) and len(v) and isinstance(v[0], str):
                v = [v]
            for e in v:
                if isinstance(e, tuple) and e and e[0] == "u":
                    return e[1]
    return None


def run(plant, el, cfg, method, ks=None, R=None, rho=None, x0=None,
        copy_plant=True):
    if copy_plant:
        plant = copy.deepcopy(plant)
    N, dt, K = plant.N, plant.dt, plant.K
    ctrl = Controller(cfg)
    Nh = cfg.ctrl.Nh
    sp, dmin = cfg.plant.spacing, cfg.plant.d_min

    freeze = bool(getattr(cfg, "freeze_structure", False))
    always_stop = bool(getattr(cfg, "always_stop", False))
    stop_fb = (getattr(cfg, "fallback_tube", "drift") == "stop")
    jf = _failing_actuator(plant)
    crit = ("e", jf - 1, jf) if (jf is not None and jf >= 1) else None
    k_fail = getattr(plant, "k_fail", None)

    x = plant.x0() if x0 is None else np.array(x0, dtype=float).copy()
    xhat = x.copy()
    s_bel = list(el.all)
    trajs = {i: ctrl.ballistic(x[i]) for i in range(N)}
    rb_hist = []
    ridx = 0
    flagged = set()
    H = dict(t=[], p=[], v=[], u=[], h=[], err=[], slack=[], nact=[],
             sel_ms=[], switch=[], infeas=0, qpfail=0, relax=0, crit_kept=-1)

    for k in range(K):
        # ---- risk update (pre-computed trace) -------------------------
        if R is not None and ridx < len(ks) and ks[ridx] == k:
            rb_hist.append(R[ridx])
            ridx += 1
            if len(rb_hist) >= cfg.risk.persistence:
                rec = np.stack(rb_hist[-cfg.risk.persistence:])
                flagged = set(np.flatnonzero((rec > rho).all(axis=0)))

        dead = [e for e in el.all if plant.is_dead(e, k)]

        # ---- does the belief structure need updating? -----------------
        need = False
        precursor_driven = (method == "prop") and not freeze
        if method == "post" or (method == "prop" and freeze):
            need = any(e in s_bel for e in dead)
        elif precursor_driven:
            risky = {el.all[j] for j in flagged if el.all[j][0] != "u"}
            need = any(e in s_bel for e in (set(dead) | risky))

        # ---- flagged-but-alive actuators : tighten, do not drop -------
        suspect = set()
        if method == "prop":
            for j in range(N):
                e = ("u", j, j)
                if el.index[e] in flagged and not plant.is_dead(e, k):
                    suspect.add(j)

        # ---- structure selection --------------------------------------
        if need:
            if precursor_driven:
                rb = np.array(rb_hist[-1]) if rb_hist else np.zeros(len(el))
                for j in range(N):
                    rb[el.index[("u", j, j)]] = 0.0
                thr = rho
            else:
                rb = np.zeros(len(el))
                thr = 2.0
            A = admissible_elements(el, rb, thr, dead=dead)
            s_new, ok, el_t, stg = select_structure(
                el, A, rb, s_bel, cfg, N, dead=dead, return_stage=True)
            H["sel_ms"].append(el_t * 1e3)
            if stg >= 1:
                H["relax"] += 1
            if ok:
                if set(s_new) != set(s_bel):
                    H["switch"].append(k * dt)
                s_bel = s_new
            else:
                H["infeas"] += 1

        if crit is not None and k_fail is not None and k == k_fail:
            H["crit_kept"] = int(crit in s_bel)

        # ---- effective structure (dead elements always removed) -------
        E_bel = [e for e in s_bel if e[0] == "e"]
        act_bel = {e[1] for e in s_bel if e[0] == "u"}
        sen_bel = {e[1] for e in s_bel if e[0] == "v"}
        adj = neighbors([e for e in E_bel if not plant.is_dead(e, k)], N)
        sen_ok = {i for i in sen_bel if not plant.is_dead(("v", i, i), k)}
        act_phys = {i for i in range(N) if not plant.is_dead(("u", i, i), k)}

        # ---- state estimation ------------------------------------------
        for i in range(N):
            if (i in sen_ok) or (adj[i] & sen_ok):
                xhat[i] = x[i]
            else:
                xhat[i, 0] += xhat[i, 1] * dt + cfg.ctrl.est_drift * dt * dt
                xhat[i, 1] *= (1.0 - cfg.plant.drag * dt)

        pref_nom = plant.p_ref(k)
        pfail = {j: np.full(Nh + 1, xhat[j, 0])
                 for j in range(N) if j not in act_bel}

        # ---- distributed MPC -------------------------------------------
        u = np.zeros(N)
        slack = 0.0
        for i in range(N):
            if i not in act_bel:
                trajs[i] = ctrl.ballistic(xhat[i])
                continue
            tt = np.arange(1, Nh + 1) * dt
            pref = pref_nom[i] + cfg.plant.v_ref * tt
            vref = np.full(Nh, cfg.plant.v_ref)
            if pfail:
                jj = min(pfail)
                pref = pfail[jj][1:] + (i - jj) * sp
                vref = np.zeros(Nh)
                for j, pj in pfail.items():
                    if j > i:
                        pref = np.minimum(pref, pj[1:] - (j - i) * sp)
            nb = {}
            for j in (i - 1, i + 1):
                if not (0 <= j < N):
                    continue
                sgn = 1.0 if j > i else -1.0
                if j not in act_bel:
                    nb[j] = (pfail[j], sgn)
                elif j in adj[i]:
                    tj = trajs[j]
                    if (j in suspect) or (always_stop and sgn > 0):
                        bal = np.full(Nh + 1, xhat[j, 0])   # immediate stop
                        tj = np.minimum(tj, bal) if sgn > 0 else np.maximum(tj, bal)
                    nb[j] = (tj, sgn)
                else:
                    if sgn > 0 and (stop_fb or always_stop):
                        nb[j] = (np.full(Nh + 1, xhat[j, 0]), sgn)   # Remark 3(i)
                    else:
                        base = ctrl.ballistic(xhat[j])               # Remark 3(iii)
                        m = 0.5 * cfg.ctrl.drift_rate * (np.arange(Nh + 1) * dt) ** 2
                        nb[j] = (base - sgn * m, sgn)
            U, sk, ok = ctrl.solve(xhat[i], pref, vref, nb)
            if not ok:
                H["qpfail"] += 1
            u[i] = U[0]
            slack = max(slack, sk)
            trajs[i] = ctrl.predict(xhat[i], U)
        u = np.array([u[i] if i in act_phys else 0.0 for i in range(N)])

        # ---- log and step ----------------------------------------------
        gaps = np.diff(x[:, 0]) - dmin
        viol = float(max(0.0, -gaps.min()))
        ferr = float(np.abs(gaps + dmin - sp).max())
        H["t"].append(k * dt)
        H["p"].append(x[:, 0].copy())
        H["v"].append(x[:, 1].copy())
        H["u"].append(u.copy())
        H["h"].append(gaps.copy())
        H["slack"].append(slack)
        H["err"].append(ferr + 2.0 * viol)
        H["nact"].append(len(act_bel))
        x = plant.step(x, u, k)

    H["hard_relaxed"] = ctrl.hard_relaxed
    for key in ("t", "p", "v", "u", "h", "err", "slack", "nact"):
        H[key] = np.array(H[key])
    return H

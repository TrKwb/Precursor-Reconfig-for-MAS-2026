# -*- coding: utf-8 -*-
"""Closed-loop runner.

The three methods differ ONLY in when the *belief* structure s_bel is updated:
  'fixed' : never; 'post' : at the instant of failure (oracle);
  'prop'  : when the conformal flag fires.
Physics, disturbances and diagnostics are identical (the plant is deep-copied),
so the comparison is paired.

Flags in cfg (all default OFF; used by exp9):
  force_critical   Remedy 2, handled in select.py
  fallback_tube    'drift'      : nominal-drift prediction (Remark 3(iii)), default
                   'stop'       : when the edge to a forward neighbour is absent from
                                  the belief structure, use the immediate-stop tube
                                  (Remark 3(i))
                   'stop_alive' : as 'stop', but only while the edge is physically
                                  alive; a DEAD edge falls back to drift
  always_stop      immediate-stop tube for EVERY forward neighbour (baseline)
  hard_barrier     two-tier constraint, handled in control.py
  freeze_structure tightening from detection, but structure changes only at
                   failure (no pre-emptive reconfiguration)

Theorem policy pi_T (exp10; all default OFF -> legacy paths are bit-identical):
  policy_T         enable pi_T.  For every FORWARD neighbour j of agent i:
                     j not actuated (dead)  -> stop tube           (as before)
                     j latched              -> stop tube, whether or not the
                                               edge (i,j) is in the structure
                     otherwise              -> braking-envelope tube (Lemma 5),
                                               or the legacy tube if
                                               unlatched_tube == 'comm'
                   plus a one-step safety shield and v_i >= 0 for every agent.
  latch_release    None (default): a fired actuator flag is held until failure;
                   int n: released after n consecutive risk evaluations unflagged.
  unlatched_tube   'env' (default) | 'comm'
  shield           default True under policy_T.  The QP input of follower i is
                   accepted only if the next state lies in the set F from which
                   maximal braking satisfies the DCBF decay against the current
                   tube (stop tube if the leader is latched or dead, otherwise the
                   leader's maximal-braking envelope); otherwise maximal braking is
                   applied.  F is invariant under maximal braking, so this enforces
                   the terminal condition (T2) without touching control.py.  With
                   the leader latched but the state not yet in F the shield brakes:
                   the reach phase of Theorem 1.
  shield_margin    added to d_min inside the shield (disturbance allowance), default 0.
  rear_constraints default True (legacy).  False drops the constraints an agent
                   imposes against its rear neighbour (pure assume-guarantee split).

Run dump (environment variables; inherited by spawned worker processes):
  MAS_DUMP         comma list of fault kinds to keep (u,v,e) or 'all'; unset = off
  MAS_DUMP_KFAIL   keep only trials with this k_fail (optional)
  MAS_DUMP_DIR     output directory (default results/dump)
  Files are named <trial>_<method>_<cond>.pkl; <trial> is identical across
  conditions of the same trial (failing elements, k_fail, initial positions).

Logged critical-edge information (actuator faults with jf >= 1 only):
  crit_in[k]   1/0 : is the critical edge e(jf-1, jf) in the belief structure used
               for control at step k (AFTER structure selection at k);
               -1 for every k when there is no critical edge
  crit_kept    crit_in[k_fail-1] : the structure in force when the failure occurs.
  tube_kind[k] tube used by agent jf-1 against jf at step k: 'comm', 'drift',
               'stop', 'stopdead', 'env', 'reach' ('' if no critical pair)
  shield[k]    number of agents whose QP input was overridden by the shield
  latched[k]   (N,) bool latch state of each actuator (all False if policy_T off)
  est_err[k]   (N,) |xhat - x| in position, before control at step k
"""
import copy
import hashlib
import os
import pickle
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


def _param(cfg, names):
    """First existing dotted attribute among names; fail loudly if none exists."""
    for nm in names:
        obj = cfg
        try:
            for part in nm.split("."):
                obj = getattr(obj, part)
            return float(obj)
        except AttributeError:
            continue
    raise AttributeError("none of %s found in cfg; add the right name" % (names,))


_DUMP_FLAGS = ("policy_T", "freeze_structure", "fallback_tube", "always_stop",
               "hard_barrier", "nonretreat", "force_critical", "latch_release",
               "unlatched_tube", "shield", "terminal_kappa")


def _maybe_dump(H, plant, cfg, method):
    """Pickle H if the environment asks for it (works in spawned workers).
    MAS_DUMP       comma list of fault kinds to keep: u,v,e or all
    MAS_DUMP_KFAIL keep only trials with this k_fail (optional)
    MAS_DUMP_DIR   output directory (default results/dump)"""
    spec = os.environ.get("MAS_DUMP")
    if not spec:
        return
    fi = plant.fail_info() if hasattr(plant, "fail_info") else {}
    fails = [tuple(e) for e in fi.get("failing", [])]
    kinds = [s.strip() for s in spec.split(",")]
    if "all" not in kinds and not any(e[0] in kinds for e in fails):
        return
    kf_only = os.environ.get("MAS_DUMP_KFAIL")
    if kf_only and int(kf_only) != fi.get("k_fail"):
        return
    flags = {nm: getattr(cfg, nm, None) for nm in _DUMP_FLAGS}
    x0 = np.asarray(H["p"][0], dtype=float)
    trial = hashlib.sha1(repr((fails, fi.get("k_fail"), np.round(x0, 9).tolist())).encode()).hexdigest()[:12]
    cond = hashlib.sha1(repr((method, sorted(flags.items()))).encode()).hexdigest()[:8]
    d = os.environ.get("MAS_DUMP_DIR", os.path.join("results", "dump"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s_%s_%s.pkl" % (trial, method, cond)), "wb") as fh:
        pickle.dump(dict(meta=dict(method=method, flags=flags, fail=fi,
                                   trial=trial, cond=cond), H=H), fh)


class _Brake:
    """Nominal model (1) under maximal admissible braking (keeps v >= 0)."""

    def __init__(self, dt, drag, umax, dmin, gamma):
        self.dt, self.a, self.umax = dt, 1.0 - drag * dt, umax
        self.dmin, self.gam = dmin, gamma

    def u_brake(self, v):
        v = max(v, 0.0)
        return -self.umax if self.a * v - self.dt * self.umax >= 0 else -self.a * v / self.dt

    def u_floor(self, v):                        # smallest input with v_next >= 0
        return max(-self.umax, -self.a * max(v, 0.0) / self.dt)

    def step(self, p, v, u):
        return p + self.dt * v + 0.5 * self.dt ** 2 * u, self.a * v + self.dt * u

    def traj(self, p, v, n):
        out = np.empty(n + 1); out[0] = p; v = max(v, 0.0)
        for t in range(n):
            p, v = self.step(p, v, self.u_brake(v)); out[t + 1] = p
        return out

    def in_F(self, gap, vf, vl=None, margin=0.0, nmax=400):
        """gap = p_leader - p_follower.  vl=None: leader frozen (stop tube);
        otherwise the leader follows its maximal-braking envelope from speed vl.
        True iff follower maximal braking keeps h >= 0 and h(t+1) >= (1-gam) h(t)."""
        dm = self.dmin + margin
        h = gap - dm
        if h < 0:
            return False
        pf, vf = 0.0, max(vf, 0.0)
        pl, vl = gap, (0.0 if vl is None else max(vl, 0.0))
        for _ in range(nmax):
            if vf <= 1e-12 and vl <= 1e-12:
                return True
            pf, vf = self.step(pf, vf, self.u_brake(vf))
            if vl > 1e-12:
                pl, vl = self.step(pl, vl, self.u_brake(vl))
            hn = pl - pf - dm
            if hn < (1.0 - self.gam) * h - 1e-12:
                return False
            h = hn
        return True


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
    fb = getattr(cfg, "fallback_tube", "drift")
    assert fb in ("drift", "stop", "stop_alive"), "unknown fallback_tube %r" % (fb,)
    stop_fb = fb in ("stop", "stop_alive")
    stop_alive_only = (fb == "stop_alive")

    # ---- pi_T options (no effect on legacy conditions) -----------------
    pol_T = bool(getattr(cfg, "policy_T", False))
    latch_release = getattr(cfg, "latch_release", None)
    unl_tube = getattr(cfg, "unlatched_tube", "env")
    assert unl_tube in ("env", "comm"), "unknown unlatched_tube %r" % (unl_tube,)
    use_shield = pol_T and bool(getattr(cfg, "shield", True))
    sh_margin = float(getattr(cfg, "shield_margin", 0.0))
    rear_cons = bool(getattr(cfg, "rear_constraints", True))
    BR = None
    if pol_T:
        assert method == "prop", "policy_T requires method 'prop'"
        BR = _Brake(dt, cfg.plant.drag,
                    _param(cfg, ("plant.u_max", "ctrl.u_max", "plant.umax", "ctrl.umax")),
                    dmin,
                    _param(cfg, ("ctrl.gamma", "ctrl.cbf_gamma", "ctrl.gam", "cbf.gamma")))

    jf = _failing_actuator(plant)
    crit = ("e", jf - 1, jf) if (jf is not None and jf >= 1) else None
    if crit is not None:
        assert crit in el.index, \
            "critical edge %s not in el.index (edge naming mismatch)" % (crit,)
    k_fail = getattr(plant, "k_fail", None)

    def _edge_dead(i, j, k):
        eij = ("e", min(i, j), max(i, j))
        return (eij in el.index) and plant.is_dead(eij, k)

    x = plant.x0() if x0 is None else np.array(x0, dtype=float).copy()
    xhat = x.copy()
    s_bel = list(el.all)
    trajs = {i: ctrl.ballistic(x[i]) for i in range(N)}
    rb_hist = []
    ridx = 0
    flagged = set()
    latched = np.zeros(N, dtype=bool)            # pi_T latch, per actuator
    quiet = np.zeros(N, dtype=int)
    H = dict(t=[], p=[], v=[], u=[], h=[], err=[], slack=[], nact=[],
             crit_in=[], sel_ms=[], switch=[], infeas=0, qpfail=0, relax=0,
             crit_kept=-1, tube_kind=[], shield=[], latched=[], est_err=[])

    for k in range(K):
        # ---- risk update (pre-computed trace) -------------------------
        new_eval = False
        if R is not None and ridx < len(ks) and ks[ridx] == k:
            rb_hist.append(R[ridx])
            ridx += 1
            if len(rb_hist) >= cfg.risk.persistence:
                rec = np.stack(rb_hist[-cfg.risk.persistence:])
                flagged = set(np.flatnonzero((rec > rho).all(axis=0)))
                new_eval = True

        dead = [e for e in el.all if plant.is_dead(e, k)]

        # ---- pi_T latch (updated once per risk evaluation) -------------
        if pol_T and new_eval:
            for j in range(N):
                if el.index[("u", j, j)] in flagged:
                    latched[j], quiet[j] = True, 0
                elif latched[j]:
                    quiet[j] += 1
                    if latch_release is not None and quiet[j] >= int(latch_release):
                        latched[j], quiet[j] = False, 0

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

        # ---- critical edge in the structure used at step k ------------
        H["crit_in"].append(int(crit in s_bel) if crit is not None else -1)

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
        H["est_err"].append(np.abs(xhat[:, 0] - x[:, 0]))

        pref_nom = plant.p_ref(k)
        pfail = {j: np.full(Nh + 1, xhat[j, 0])
                 for j in range(N) if j not in act_bel}

        # ---- distributed MPC -------------------------------------------
        u = np.zeros(N)
        slack = 0.0
        n_shield = 0
        tk_crit = ""
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
            fwd_kind = None                        # tube kind used against i+1
            for j in (i - 1, i + 1):
                if not (0 <= j < N):
                    continue
                sgn = 1.0 if j > i else -1.0
                if pol_T and sgn < 0 and not rear_cons:
                    continue
                if j not in act_bel:
                    nb[j] = (pfail[j], sgn)
                    kind = "stopdead"
                elif pol_T and sgn > 0 and latched[j]:
                    nb[j] = (np.full(Nh + 1, xhat[j, 0]), sgn)          # stop tube, no link needed
                    kind = "stop"
                elif pol_T and sgn > 0 and unl_tube == "env":
                    nb[j] = (BR.traj(xhat[j, 0], xhat[j, 1], Nh), sgn)  # envelope tube, Lemma 5
                    kind = "env"
                elif j in adj[i]:
                    tj = trajs[j]
                    kind = "comm"
                    if (j in suspect) or (always_stop and sgn > 0):
                        bal = np.full(Nh + 1, xhat[j, 0])   # immediate stop
                        tj = np.minimum(tj, bal) if sgn > 0 else np.maximum(tj, bal)
                        kind = "stop"
                    nb[j] = (tj, sgn)
                else:
                    use_stop = always_stop or (
                        stop_fb and not (stop_alive_only and _edge_dead(i, j, k)))
                    if sgn > 0 and use_stop:
                        nb[j] = (np.full(Nh + 1, xhat[j, 0]), sgn)   # Remark 3(i)
                        kind = "stop"
                    else:
                        base = ctrl.ballistic(xhat[j])               # Remark 3(iii)
                        m = 0.5 * cfg.ctrl.drift_rate * (np.arange(Nh + 1) * dt) ** 2
                        nb[j] = (base - sgn * m, sgn)
                        kind = "drift"
                if sgn > 0:
                    fwd_kind = kind
            U, sk, ok = ctrl.solve(xhat[i], pref, vref, nb)
            if not ok:
                H["qpfail"] += 1

            if pol_T:
                U = np.array(U, dtype=float)
                U[0] = max(U[0], BR.u_floor(xhat[i, 1]))           # v_i >= 0 (Assumption 2)
                if use_shield and i + 1 < N:
                    j = i + 1
                    frozen = (j not in act_bel) or bool(latched[j])
                    p1, v1 = BR.step(xhat[i, 0], xhat[i, 1], U[0])
                    if frozen:
                        ok_next = BR.in_F(xhat[j, 0] - p1, v1, None, sh_margin)
                    else:
                        pl1, vl1 = BR.step(xhat[j, 0], max(xhat[j, 1], 0.0),
                                           BR.u_brake(xhat[j, 1]))
                        ok_next = BR.in_F(pl1 - p1, v1, vl1, sh_margin)
                    if not ok_next:
                        U[0] = BR.u_brake(xhat[i, 1])
                        n_shield += 1
                        if frozen and fwd_kind == "stop":
                            fwd_kind = "reach"
            if crit is not None and i == jf - 1:
                tk_crit = fwd_kind or ""

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
        H["tube_kind"].append(tk_crit)
        H["shield"].append(n_shield)
        H["latched"].append(latched.copy())
        x = plant.step(x, u, k)

    H["hard_relaxed"] = ctrl.hard_relaxed
    for key in ("t", "p", "v", "u", "h", "err", "slack", "nact", "crit_in",
                "tube_kind", "shield", "latched", "est_err"):
        H[key] = np.array(H[key])
    # structure in force when the failure occurs (before the post-failure re-selection)
    if crit is not None and k_fail is not None and 1 <= k_fail <= K:
        H["crit_kept"] = int(H["crit_in"][k_fail - 1])
    H["nonretreat_relaxed"] = getattr(ctrl, "nonretreat_relaxed", 0)
    _maybe_dump(H, plant, cfg, method)
    return H
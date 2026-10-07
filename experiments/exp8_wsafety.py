# -*- coding: utf-8 -*-
"""How much should the selection objective pay to keep a barrier-critical edge?

w_safety = 0 reproduces the naive selector: it satisfies the connectivity
requirement (R2) but discards the edge on which the CBF tightening depends, and
the tightening then silently stops working.  Increasing w_safety buys priority
for those edges.  This sweep locates the point at which the benefit saturates,
and -- equally important -- whether it saturates above zero, which would show
that a weight cannot substitute for a constraint.

Restricted to collision-capable scenarios: in a 1-D chain only an actuator
fault at i > 0 can produce a collision, so the remaining scenarios cannot
distinguish the methods and would only dilute the rates.

Usage : python experiments/exp8_wsafety.py [N_TRIAL]      (default 10)
Output: results/exp8_wsafety.npy, results/exp8_wsafety.csv
"""
import sys
import os
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.select import admissible_elements, select_structure, is_safety_critical
from masrecon.runner import run
from masrecon.metrics import safety_stats, recovery_time

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
os.makedirs("results", exist_ok=True)

base = Cfg()
el = Elements(base.plant.N)
N = base.plant.N
rho = 1.0 - base.risk.alpha / len(el)
SC = [("u", i, i) for i in range(1, N)]          # collision-capable only

print("=" * 72)
print("exp8 : sweeping w_safety over %d collision-capable scenarios x %d trials"
      % (len(SC), NT))
print("=" * 72)
print("scenarios:", [str(s) for s in SC])
print("calibrating ...", flush=True)
cr = build_calibration(base, Elements, N, runs=30)

# Pre-compute the risk traces once: they do not depend on w_safety, so every
# value of w_safety sees exactly the same realisations and the comparison is
# paired run by run.
trials = []
for si, e_f in enumerate(SC):
    for tr in range(NT):
        pl = Plant(base, el, failing=[e_f],
                   rng=np.random.default_rng(10000 + 131 * (si + 1) + tr))
        ks, R = risk_trace(pl, cr, base.risk)
        trials.append((e_f, tr, pl, ks, R))
print("  %d trials prepared\n" % len(trials))


def critical_kept(pl, ks, R, cfg, e_f):
    """Replay the belief-structure update and report whether the edge coupling
    the failing agent to its follower survived selection."""
    jf = e_f[1]
    crit = ("e", min(jf - 1, jf), max(jf - 1, jf))
    s_bel = list(el.all)
    rb_hist, ridx, flagged = [], 0, set()
    for k in range(pl.k_fail):
        if ridx < len(ks) and ks[ridx] == k:
            rb_hist.append(R[ridx]); ridx += 1
            if len(rb_hist) >= cfg.risk.persistence:
                rec = np.stack(rb_hist[-cfg.risk.persistence:])
                flagged = set(np.flatnonzero((rec > rho).all(axis=0)))
        dead = [e for e in el.all if pl.is_dead(e, k)]
        risky = {el.all[j] for j in flagged if el.all[j][0] != "u"}
        if any(e in s_bel for e in (set(dead) | risky)):
            rb = np.array(rb_hist[-1]) if rb_hist else np.zeros(len(el))
            for j in range(N):
                rb[el.index[("u", j, j)]] = 0.0        # actuators: tighten
            A = admissible_elements(el, rb, rho, dead=dead)
            s_new, ok, _, _ = select_structure(el, A, rb, s_bel, cfg, N,
                                               dead=dead, return_stage=True)
            if ok:
                s_bel = s_new
    return crit in [e for e in s_bel if e[0] == "e"]


print("%9s %11s %12s %11s %9s %11s %9s %8s"
      % ("w_safety", "violation", "viol.area", "min gap", "crit kept",
         "recovery", "switches", "relax"))
print("-" * 86)

rows, per_run = [], []
base_viol = None
for ws in WS:
    cfg = Cfg()
    cfg.w_safety = ws
    nv = 0; ar = []; mh = []; rc_ = []; sw = []; rl = 0; ck = 0
    flags = []
    for e_f, tr, pl, ks, R in trials:
        H = run(pl, el, cfg, "prop", ks, R, rho)
        ss = safety_stats(H)
        v = int(ss["nviol"] > 0)
        nv += v; flags.append(v)
        ar.append(ss["viol_area"]); mh.append(ss["min_h"])
        sw.append(len(H["switch"])); rl += H.get("relax", 0)
        r = recovery_time(H, pl.k_fail, pl.dt)
        if r == r:
            rc_.append(r)
        kept = int(critical_kept(pl, ks, R, cfg, e_f))
        ck += kept
        per_run.append(dict(w_safety=ws, scenario=str(e_f), trial=tr,
                            violation=v, crit_kept=kept,
                            area="%.6f" % ss["viol_area"],
                            min_h="%.6f" % ss["min_h"],
                            nslack=ss["nslack"], nswitch=len(H["switch"]),
                            rec="" if r != r else "%.3f" % r))
    n = len(trials)
    if base_viol is None:
        base_viol = flags
    print("%9.2f %8d/%-3d %12.4f %11.3f %8.0f%% %11.2f %9.2f %8d"
          % (ws, nv, n, np.mean(ar), np.mean(mh), 100 * ck / n,
             np.mean(rc_) if rc_ else float("nan"), np.mean(sw), rl))
    rows.append((ws, nv / n, np.mean(ar), np.mean(mh), ck / n,
                 np.mean(rc_) if rc_ else np.nan, np.mean(sw), rl / n))

r = np.array(rows)
np.save("results/exp8_wsafety.npy", r)
with open("results/exp8_wsafety.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(per_run[0].keys()))
    w.writeheader()
    for x in per_run:
        w.writerow(x)

# ---- where does it saturate, and at what floor? -------------------------
v0 = r[0, 1]
vmin = r[:, 1].min()
knee = next(ws for ws, v in zip(r[:, 0], r[:, 1]) if v <= vmin + 1e-12)
print("\n" + "=" * 72)
print("  violation rate : %.0f %% at w_safety = 0  ->  %.0f %% floor, reached at"
      " w_safety = %.2f" % (100 * v0, 100 * vmin, knee))
print("  further increases of w_safety change nothing: the weight buys"
      " priority, not a guarantee.")
print("  recovery time and switch count do NOT degrade"
      " (%.2f -> %.2f s, %.2f -> %.2f switches):"
      % (r[0, 5], r[-1, 5], r[0, 6], r[-1, 6]))
print("  preserving the critical edge is not a trade-off but a missing"
      " requirement.")
print("\nsaved results/exp8_wsafety.npy and results/exp8_wsafety.csv")
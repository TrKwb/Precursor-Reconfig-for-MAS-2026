# -*- coding: utf-8 -*-
"""exp11b: state-based check of Theorem 1 and which pair violated.
E_s : latched continuously up to k_f AND follower state in F_stop at k_f  (theorem hypothesis)
E_L : latched on [k_f-L, k_f]  for L in --lstars
Usage: python -X utf8 experiments/exp11b_check.py [--leads real,0.4,1]"""
import sys, os, copy, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run

ap = argparse.ArgumentParser()
ap.add_argument("--leads", default="real,0.4,1")
ap.add_argument("--lstars", default="2,7,14")
ap.add_argument("--max-runs", type=int, default=0)
args = ap.parse_args()
LSTARS = [int(s) for s in args.lstars.split(",")]
GAMMA, UM, CDRAG = 0.25, 3.6, 1.0          # CHECK: same as config (gamma, u_max, c)

TP = dict(w_safety=2.0, policy_T=True, nonretreat=True, hard_barrier=True)
CONDS = [("T_latch_stop",   dict(TP, freeze_structure=True)),
         ("T_latch_rel10",  dict(TP, freeze_structure=True, latch_release=10)),
         ("T_latch_reconf", dict(TP, force_critical=True, fallback_tube="stop")),
         ("T_rel10_prefail", dict(TP, freeze_structure=True, latch_release=10,
                                  nonretreat_scope="prefail"))]   # recovery after a miss?

base = Cfg(); N = base.plant.N; el = Elements(N); rc = base.risk
rho = 1.0 - rc.alpha / len(el); dt = base.plant.dt; DMIN = base.plant.d_min

def brake(v):
    u = max(-UM, -(1 - CDRAG * dt) * v / dt)
    return dt * v + 0.5 * dt * dt * u, (1 - CDRAG * dt) * v + dt * u

def in_Fstop(h, v):            # follower max braking vs stopped predecessor: h>=0 and DCBF decay
    for _ in range(40):
        s, v = brake(max(v, 0.0)); h2 = h - s
        if h2 < -1e-9 or h2 < (1 - GAMMA) * h - 1e-9: return False
        h = h2
        if v <= 1e-9: return True
    return True

def mk(**over):
    c = Cfg()
    for k, v in over.items(): setattr(c, k, v)
    return c

def first_flag(ks, R, j):
    cnt = 0
    for i, a in enumerate(np.asarray(R)[:, j] > rho):
        cnt = cnt + 1 if a else 0
        if cnt >= rc.persistence: return int(ks[i])
    return -1

cr = build_calibration(base, Elements, N, runs=30)
runs = [("%s/%d" % (e, tr), e, 10000 + 131 * (si + 1) + tr)
        for si, e in enumerate([("u", i, i) for i in range(1, min(N, 4))]) for tr in range(10)]
if args.max_runs: runs = runs[:args.max_runs]

rows = []
for ri, (lab, e_f, seed) in enumerate(runs):
    pl = Plant(mk(), el, failing=[e_f], rng=np.random.default_rng(seed))
    ks, R = risk_trace(pl, cr, rc); ks = np.asarray(ks); R = np.asarray(R, float)
    x0 = pl.x0(); kf = pl.k_fail; jf = e_f[1]; i_f = jf - 1; idx = el.index[e_f]
    for lead in [s.strip() for s in args.leads.split(",")]:
        Rs = R.copy()
        if lead != "real":
            Rs[:, idx] = 0.0
            if lead != "none": Rs[ks >= kf - int(round(float(lead) / dt)), idx] = 1.0
        kd = first_flag(ks, Rs, idx)
        for name, over in CONDS:
            H = run(copy.deepcopy(pl), el, mk(**over), "prop", ks, Rs, rho, x0=x0)
            h = np.asarray(H["h"], float); lat = np.asarray(H["latched"]).astype(bool)
            V = np.asarray(H["v"], float) if "v" in H else None
            vf = (lambda k: float(V[k, i_f])) if (V is not None and V.ndim == 2) else \
                 (lambda k: float((np.asarray(H["p"])[k + 1, i_f] - np.asarray(H["p"])[k, i_f]) / dt))
            col = lat[:kf + 1, jf]
            llen = 0 if not col[-1] else (len(col) - 1 - (np.flatnonzero(~col)[-1] if (~col).any() else -1))
            kl = kf - llen + 1
            Es = llen > 0 and in_Fstop(h[kf, i_f], vf(kf))
            reach = None
            if llen > 0:
                for k in range(kl, kf + 1):
                    if in_Fstop(h[k, i_f], vf(k)): reach = k - kl; break
            vpairs = [int(p) for p in np.flatnonzero((h < 0).any(axis=0))]
            rows.append(dict(cond=name, lead=lead, run=lab, viol=bool(vpairs),
                             vfail=bool((h[:, i_f] < 0).any()), vpairs=vpairs, llen=int(llen),
                             Es=bool(Es), reach=reach, h_kl=float(h[kl, i_f]) if llen else np.nan,
                             v_kl=vf(kl) if llen else np.nan,
                             nr_end=bool(h[-1].min() < 0)))
    print("  %d/%d runs" % (ri + 1, len(runs)), flush=True)

print("\nN=%d  failing pair = (jf-1, jf).  E_s = latched to k_f and follower in F_stop at k_f" % N)
print("%-16s %5s %6s %8s %9s  %s" % ("cond", "lead", "viol", "failpair", "E_s/vInE",
      "  ".join("E_L%d/vInE" % L for L in LSTARS)))
for cnd, _ in CONDS:
    for lead in sorted({r["lead"] for r in rows}):
        S = [r for r in rows if r["cond"] == cnd and r["lead"] == lead]
        if not S: continue
        es = "%d/%d" % (sum(r["Es"] for r in S), sum(r["Es"] and r["vfail"] for r in S))
        el_ = "  ".join("%9s" % ("%d/%d" % (sum(r["llen"] >= L + 1 for r in S),
                        sum(r["llen"] >= L + 1 and r["vfail"] for r in S))) for L in LSTARS)
        rc_ = [r["reach"] for r in S if r["reach"] is not None]
        print("%-16s %5s %6s %8d %9s  %s   reach[steps] med %s max %s   stuck-in-violation at T %d" % (
            cnd, lead, "%d/%d" % (sum(r["viol"] for r in S), len(S)), sum(r["vfail"] for r in S),
            es, el_, np.median(rc_) if rc_ else "-", max(rc_) if rc_ else "-",
            sum(r["nr_end"] for r in S)))
print("\nviolating runs with a latch before k_f:")
for r in rows:
    if r["viol"] and r["llen"] > 0:
        print("  %-15s lead %-4s %-16s latched %2d steps  reach %s  in F_stop@kf %s  h(kl)=%.3f v(kl)=%.2f  pairs %s" % (
            r["cond"], r["lead"], r["run"], r["llen"], r["reach"], r["Es"], r["h_kl"], r["v_kl"], r["vpairs"]))
bad = [r for r in rows if r["Es"] and r["vfail"]]
print("\nTHEOREM COUNTEREXAMPLES (E_s and failing pair violated, must be empty): %d" % len(bad))
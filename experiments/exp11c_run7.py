# -*- coding: utf-8 -*-
"""Why was the gap margin only 0.22 m at latch onset in run ('u',2,2)/7, oracle lead 0.4 s?
Usage: python -X utf8 experiments/exp11c_run7.py"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run

base = Cfg(); N = base.plant.N; el = Elements(N); rc = base.risk
rho = 1.0 - rc.alpha / len(el); dt = base.plant.dt
cr = build_calibration(base, Elements, N, runs=30)
e_f = ("u", 2, 2); tr = 7; seed = 10000 + 131 * 2 + tr; LEAD = 0.4
jf, i_f = 2, 1
pl = Plant(Cfg(), el, failing=[e_f], rng=np.random.default_rng(seed))
ks, R = risk_trace(pl, cr, rc); ks = np.asarray(ks); R = np.asarray(R, float)
x0 = pl.x0(); kf = pl.k_fail; kdeg = pl.k_deg; idx = el.index[e_f]
Rs = R.copy(); Rs[:, idx] = 0.0; Rs[ks >= kf - int(round(LEAD / dt)), idx] = 1.0
inv = {v: k for k, v in el.index.items()}
print("run %s/%d seed %d  k_deg %d  k_fail %d  rho %.4f" % (e_f, tr, seed, kdeg, kf, rho))
print("\nelements above rho before k_fail+20 (evaluation steps):")
for j in range(Rs.shape[1]):
    hit = ks[(Rs[:, j] > rho) & (ks <= kf + 20)]
    if len(hit): print("  %-14s %s" % (inv[j], list(hit[:25])))

def segs(col):
    out, s = [], None
    for k, a in enumerate(col):
        if a and s is None: s = k
        if not a and s is not None: out.append((s, k - 1)); s = None
    if s is not None: out.append((s, len(col) - 1))
    return out

TP = dict(w_safety=2.0, policy_T=True, nonretreat=True, hard_barrier=True)
CONDS = [("T_latch_stop", dict(TP, freeze_structure=True)),
         ("T_latch_rel10", dict(TP, freeze_structure=True, latch_release=10)),
         ("T_latch_reconf", dict(TP, force_critical=True, fallback_tube="stop"))]
for name, over in CONDS:
    c = Cfg()
    for k, v in over.items(): setattr(c, k, v)
    H = run(copy.deepcopy(pl), el, c, "prop", ks, Rs, rho, x0=x0)
    h = np.asarray(H["h"], float); lat = np.asarray(H["latched"]).astype(bool)
    V = np.asarray(H["v"], float) if "v" in H else np.diff(np.asarray(H["p"], float), axis=0) / dt
    sh = np.asarray(H["shield"])
    print("\n" + "=" * 70 + "\n%s   violated pairs: %s   min h_failpair %.3f" % (
        name, list(np.flatnonzero((h < 0).any(axis=0))), h[:, i_f].min()))
    for j in range(lat.shape[1]):
        s = segs(lat[:, j])
        if s: print("  agent %d latched (its follower uses stop) on steps %s" % (j, s))
    print("  switches:", H.get("switch"))
    print("   k     t  | h per pair            | v per agent                 | latched | shield")
    for k in range(max(0, kf - 45), min(len(h), kf + 8)):
        shk = sh[k] if sh.ndim == 1 else "".join(str(int(x > 0)) for x in sh[k])
        print("%5d %5.2f | %s | %s |  %s   | %s" % (
            k, k * dt, " ".join("%6.3f" % x for x in h[k]),
            " ".join("%5.2f" % x for x in V[min(k, len(V) - 1)]),
            "".join(str(int(x)) for x in lat[k]), shk))
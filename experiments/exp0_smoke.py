# -*- coding: utf-8 -*-
#"""Smoke test: one trial, three methods.  RUN THIS FIRST.
#
#Verifies that every module works end-to-end and saves the time histories needed#by Fig. 2 (results/smoke_*.npz).
#
#Usage:  python experiments/exp0_smoke.py
#Time :  about 30 s
#"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements, phi
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
from masrecon.metrics import recovery_time, safety_stats, detection_stats

os.makedirs("results", exist_ok=True)
cfg = Cfg()
rc = cfg.risk
el = Elements(cfg.plant.N)
rho = 1.0 - rc.alpha / len(el)

print("|C| = %d  (E=%d V=%d U=%d)   Phi(all) = %s"
      % (len(el), len(el.E), len(el.V), len(el.U),
         phi(el.all, cfg.plant.N, cfg.F_max)))

t0 = time.time()
cr = build_calibration(cfg, Elements, cfg.plant.N, runs=30)
print("calibration : n=%d  rbar_max=%.5f  rho=%.5f  alpha_e=%.5f  (%.1f s)"
      % (cr.n_cal, cr.rbar_max, rho, rc.alpha / len(el), time.time() - t0))
assert cr.rbar_max > rho, \
    "n_cal too small for alpha=%.3f: need n >= %d" % (rc.alpha,
                                                      int(np.ceil(len(el) / rc.alpha)) - 1)

e_f = ("u", 2, 2)
pl = Plant(cfg, el, failing=[e_f], rng=np.random.default_rng(42))
ks, R = risk_trace(pl, cr, rc)
d = detection_stats(ks, R, el.index[e_f], rho, rc.persistence,
                    pl.k_fail, pl.dt, len(el))
print("failing element %s   t_deg=%.1f s   t_fail=%.1f s"
      % (str(e_f), pl.k_deg * pl.dt, pl.k_fail * pl.dt))
print("detection   : detected=%s  lead=%.2f s  FAR_check=%.4f"
      % (d["detected"], d["lead"], d["far_check"]))

print("\n%-8s %10s %10s %7s %7s %7s  %s"
      % ("method", "recov[s]", "min_h[m]", "nviol", "nslack", "qpfail", "switch@[s]"))
out = {}
for m in ("fixed", "post", "prop"):
    t0 = time.time()
    H = run(pl, el, cfg, m, ks, R, rho)
    ss = safety_stats(H)
    rt = recovery_time(H, pl.k_fail, pl.dt)
    out[m] = rt
    print("%-8s %10s %10.3f %7d %7d %7d  %s"
          % (m, ("%.2f" % rt) if rt == rt else "n/a", ss["min_h"], ss["nviol"],
             ss["nslack"], H["qpfail"], [round(z, 2) for z in H["switch"]]))
    np.savez("results/smoke_%s.npz" % m,
             **{k: v for k, v in H.items() if isinstance(v, np.ndarray)})

rp, rq = out["post"], out["prop"]
if rp == rp and rq == rq and rp > 0:
    print("\nrecovery-time reduction (prop vs post) : %.1f %%" % (100 * (rp - rq) / rp))

np.savez("results/smoke_risk.npz", ks=ks, R=R, rho=rho,
         k_fail=pl.k_fail, dt=pl.dt, jf=el.index[e_f])
print("saved results/smoke_*.npz  ->  Fig. 2 can now be generated")




####################
#assert cr.rbar_max > rho：α_e ≥1/(n_cal +1) が満たされないとどんなに異常でも
#フラグが立ちません。必要な n_calを明示したエラーメッセージ付きで守っています
#3手法で同一の pl と (ks, R) を共有：診断信号は外生なので、リスクトレースを1回
#だけ計算して使い回します。これが「対応のある比較」を保証します
#smoke_risk.npz に jf（故障要素のインデックス）を保存：Fig.2(a) で故障要素を
#太線、他13要素を薄いグレーで描き分けるために必要です
###################
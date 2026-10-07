# -*- coding: utf-8 -*-
"""Validity and power of the conformal detector (Figure 8), with counts.

For W in {64, 96} and alpha in {0.05, 0.20, 0.50} at N = 4, over every
single-element fault (|C| = 14 scenarios) x NSEED seeds, records
  raw  : #evaluations with r_e > rho_e on HEALTHY elements   -- Proposition 1
  flag : #evaluations at which the n_p persistence rule holds -- what the
         selector acts on (always <= raw)
  det  : #runs in which the failing element is flagged before failure
together with the denominators, so that Clopper-Pearson intervals can be drawn.
The risk trace does not depend on alpha, so it is computed once per (W, run)
and thresholded at every alpha.

Usage : python experiments/exp3b_conformal.py [NSEED]     (default 10)
Output: results/exp3b_conformal.npz (+ .sha256)
"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from _util import persistence_flags, clopper_pearson, write_hash, env_info

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WS = (64, 96)
ALPHAS = (0.05, 0.20, 0.50)
os.makedirs("results", exist_ok=True)


def set_window(cfg, W):
    for name in ("W", "window", "win", "wlen", "window_len"):
        if hasattr(cfg.risk, name):
            setattr(cfg.risk, name, W)
            return name
    raise AttributeError("no window-length attribute in cfg.risk; has: %s"
                         % [a for a in dir(cfg.risk) if not a.startswith("_")])


shape = (len(WS), len(ALPHAS))
raw_k, raw_n, flg_k, flg_n, det_k, det_n = (np.zeros(shape, np.int64) for _ in range(6))
print(env_info())
for wi, W in enumerate(WS):
    cfg = Cfg(); N = cfg.plant.N
    print("W = %d  (attribute cfg.risk.%s)" % (W, set_window(cfg, W)), flush=True)
    el = Elements(N)
    npers = cfg.risk.persistence
    cr = build_calibration(cfg, Elements, N, runs=30)
    for si, e_f in enumerate(el.all):
        for tr in range(NSEED):
            pl = Plant(cfg, el, failing=[e_f],
                       rng=np.random.default_rng(30000 + 97 * si + tr))
            ks, R = risk_trace(pl, cr, cfg.risk)
            ks = np.asarray(ks); R = np.asarray(R)
            jf = el.index[e_f]
            hl = [j for j in range(len(el)) if j != jf]
            for ai, a in enumerate(ALPHAS):
                rho = 1.0 - a / len(el)
                over = R > rho
                fl = persistence_flags(over, npers)
                raw_k[wi, ai] += over[:, hl].sum(); raw_n[wi, ai] += over[:, hl].size
                flg_k[wi, ai] += fl[npers - 1:, hl].sum(); flg_n[wi, ai] += fl[npers - 1:, hl].size
                det_k[wi, ai] += bool(np.any(fl[:, jf] & (ks < pl.k_fail))); det_n[wi, ai] += 1
    for ai, a in enumerate(ALPHAS):
        ae = a / len(el)
        lo, hi = clopper_pearson(raw_k[wi, ai], raw_n[wi, ai])
        print("  alpha=%.2f a_e=%.4f  raw %.4f [%.4f,%.4f] n=%d  %s | flag %.4f | det %d/%d"
              % (a, ae, raw_k[wi, ai] / raw_n[wi, ai], lo, hi, raw_n[wi, ai],
                 "VIOLATES" if lo > ae else ("ok" if hi < ae else "inconclusive"),
                 flg_k[wi, ai] / flg_n[wi, ai], det_k[wi, ai], det_n[wi, ai]))
np.savez("results/exp3b_conformal.npz", W=np.array(WS), alpha=np.array(ALPHAS),
         nC=len(el), npers=npers, raw_k=raw_k, raw_n=raw_n, flag_k=flg_k,
         flag_n=flg_n, det_k=det_k, det_n=det_n, env=np.array(env_info()))
print("saved results/exp3b_conformal.npz  sha256", write_hash("results/exp3b_conformal.npz")[:16])
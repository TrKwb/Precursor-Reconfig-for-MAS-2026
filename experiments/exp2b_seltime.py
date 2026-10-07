# -*- coding: utf-8 -*-
"""Structure-selection time vs |C| (Figure 6), measured properly.

* 400 random risk profiles per size; the first 5 calls per size are executed
  and DISCARDED (first-call / import / cache / allocator warm-up).
* Reported: median and [p5, p95].  The mean is not reported: it is dominated
  by rare interpreter pauses and was the source of the "mean above p95"
  artefact in the earlier figure.
* The environment string (Python / NumPy / OS / CPU) is stored with the data.

Usage : python experiments/exp2b_seltime.py
Output: results/exp2b_seltime.npz (+ .sha256)
"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from _util import time_selection, write_hash, env_info

NLIST = (4, 8, 16, 24, 32, 48, 64)       # |C| = 14, 44, 96, 144, 192, 288, 384 expected
NPROF, WARM = 400, 5
os.makedirs("results", exist_ok=True)
print(env_info())
sizes, T = [], []
for N in NLIST:
    cfg = Cfg(); cfg.plant.N = N
    el = Elements(N)
    rho = 1.0 - cfg.risk.alpha / len(el)
    t = time_selection(el, cfg, N, rho, np.random.default_rng(4242 + N), NPROF, WARM)
    sizes.append(len(el)); T.append(t)
    p5, p50, p95 = np.percentile(t, [5, 50, 95])
    print("N=%3d |C|=%4d  median %.4f ms  [p5 %.4f, p95 %.4f]  max %.3f"
          % (N, len(el), p50, p5, p95, t.max()))
np.savez("results/exp2b_seltime.npz", N=np.array(NLIST), nC=np.array(sizes),
         t_ms=np.array(T), warm=WARM, env=np.array(env_info()))
print("saved results/exp2b_seltime.npz  sha256", write_hash("results/exp2b_seltime.npz")[:16])
# -*- coding: utf-8 -*-
"""Experiment B : does the detector generalise beyond the degradation
mechanism that the feature set trivially mirrors?

'ar'    : AR(1) coefficient and variance both ramp  -> AC1 and log-sd driven directly
'bias'  : bias ramp        -> AC1, variance UNCHANGED
'osc'   : emerging oscillation, variance rescaled to be constant
'quant' : progressive quantisation / stiction

Calibration is always performed on healthy 'ar' data, i.e. the detector is NOT
retuned for each mechanism.

Usage:  python experiments/exp5_mechanism.py [N_TRIAL]     (default 30)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace, features_many
from masrecon.metrics import detection_stats, summarize

NT = int(sys.argv[1]) if len(sys.argv) > 1 else 30
os.makedirs("results", exist_ok=True)
cfg = Cfg(); rc = cfg.risk; el = Elements(4)
rho = 1.0 - rc.alpha / len(el)
FE = [("u", 2, 2), ("v", 1, 1), ("e", 0, 1), ("u", 0, 0), ("v", 3, 3), ("e", 1, 2)]

print("calibrating on HEALTHY 'ar' data (detector is not retuned per mechanism)")
cr = build_calibration(cfg, Elements, 4, runs=40)
print("  n_cal = %d   rho = %.5f   alpha_e = %.5f\n" % (cr.n_cal, rho, rc.alpha / len(el)))

print("%-8s %9s %10s %10s %10s %26s"
      % ("mech.", "detect", "lead[s]", "FAR_chk", "FAR_trial", "mean feature shift"))
print("-" * 82)
rows = []
for mech in ("ar", "bias", "osc", "quant"):
    rs = []; dF = []
    t0 = time.time()
    for t in range(NT):
        e_f = FE[t % len(FE)]
        pl = Plant(cfg, el, failing=[e_f], rng=np.random.default_rng(424000 + t))
        pl.set_mechanism(mech)
        ks, R = risk_trace(pl, cr, rc)
        rs.append(detection_stats(ks, R, el.index[e_f], rho, rc.persistence,
                                  pl.k_fail, pl.dt, len(el)))
        j = el.index[e_f]
        Fh = features_many([pl.y[k - rc.W:k, j]
                            for k in range(rc.W, pl.k_deg, rc.stride)], rc)
        Fd = features_many([pl.y[k - rc.W:k, j]
                            for k in range(pl.k_fail - 3 * rc.W, pl.k_fail, rc.stride)], rc)
        dF.append(Fd.mean(0) - Fh.mean(0))
    s = summarize(rs); dF = np.mean(dF, axis=0)
    print("%-8s %8.1f%% %10.2f %10.4f %10.2f   PE%+.2f DET%+.2f LAM%+.2f sd%+.2f AC1%+.2f"
          % (mech, 100 * s["detected"][0], s["lead"][0], s["far_check"][0],
             s["far_trial"][0], *dF))
    rows.append([("ar", "bias", "osc", "quant").index(mech),
                 s["detected"][0], s["lead"][0], s["far_check"][0], *dF])

np.save("results/exp5_mechanism.npy", np.array(rows))
print("""
Reading the feature-shift columns:
  'ar'    should show large sd and AC1 shifts  (the trivial case),
  'bias'  should show sd ~ 0 and AC1 ~ 0       (detector must use PE/DET/LAM),
  'osc'   should show sd ~ 0 with PE/DET moving,
  'quant' should show PE decreasing.
A detection rate well above zero for bias/osc/quant establishes that the result
is not an artefact of the feature set mirroring the generative parameters.""")
print("\nsaved results/exp5_mechanism.npy")
# -*- coding: utf-8 -*-
"""Main experiment : 20 fault scenarios x N_TRIAL seeds x 3 methods.

Produces Table 1 of the paper and results/exp1.npz (input for Fig. 3).

Two points of experimental hygiene.

(1) Shared initial condition and realisation.  The initial condition is drawn
    once and passed to all three methods, and each method runs on its own deep
    copy of the plant, so that any random numbers consumed inside run()
    (disturbances etc.) are identical across methods as well.  Without the
    copy, the three methods would see successively advanced generator states
    and the paired tests would, strictly, be invalid.

(2) Collision-capable subset.  In a 1-D chain only an actuator fault at i > 0
    can produce a collision (agent 0 is rearmost), so only those runs can
    distinguish the methods; a rate over all scenarios measures the
    denominator.  Both are reported, the diluted one labelled as such.

Reports the paired Wilcoxon test on recovery time, the exact McNemar test on
violation incidence, and the frequency with which the graduated fallback of
select_structure() had to relax the admissible set.

Usage:  python experiments/exp1_main.py [N_TRIAL]      (default 10)
Time :  a few minutes for N_TRIAL = 10 on one core
"""
import sys
import os
import time
import math
import copy
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
from masrecon.metrics import recovery_time, safety_stats, detection_stats, summarize

try:
    from masrecon.select import FALLBACK
except ImportError:
    FALLBACK = "n/a"

N_TRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 10
os.makedirs("results", exist_ok=True)
METHODS = ("fixed", "post", "prop")
OUT = "results/exp1.npz"
BACKUP = "results/exp1_pre_x0fix.npz"


def mcnemar(a, b):
    """Paired exact McNemar.  a, b boolean (True = violation).
    Returns (a-only, b-only, p)."""
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10
    if n == 0:
        return b10, b01, 1.0
    k = min(b01, b10)
    return b10, b01, min(2.0 * sum(math.comb(n, i)
                                   for i in range(k + 1)) / 2.0 ** n, 1.0)


def wilcoxon(x, y):
    """Paired Wilcoxon signed-rank, normal approximation with tie correction.
    Returns (n_pairs, y_faster, y_slower, p)."""
    d = np.asarray(y) - np.asarray(x)
    d = d[~np.isnan(d)]
    nz = d[d != 0]
    n = len(nz)
    if n < 6:
        return len(d), int((d < 0).sum()), int((d > 0).sum()), float("nan")
    r = np.empty(n)
    o = np.argsort(np.abs(nz))
    a = np.abs(nz)[o]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w = min(r[nz > 0].sum(), r[nz < 0].sum())
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    return (len(d), int((d < 0).sum()), int((d > 0).sum()),
            math.erfc(abs((w - mu + 0.5) / sd) / math.sqrt(2.0)))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    dd = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return max(0.0, c - h), min(1.0, c + h)


def collision_capable(e_f):
    """Only a stopped actuator with a follower behind it can be struck."""
    return e_f[0] == "u" and e_f[1] > 0


def braking_distance(cfg):
    c, v0, um = cfg.plant.drag, cfg.plant.v_ref, cfg.plant.u_max
    ts = math.log((v0 + um / c) / (um / c)) / c
    return (v0 + um / c) * (1 - math.exp(-c * ts)) / c - um * ts / c, ts


# ------------------------------------------------------------------ setup
cfg = Cfg()
rc = cfg.risk
el = Elements(cfg.plant.N)
rho = 1.0 - rc.alpha / len(el)

SC = ([("u", i, i) for i in range(4)]              # 4 actuator faults
      + [("v", i, i) for i in range(4)]            # 4 sensor faults
      + list(el.E)                                 # 6 link faults
      + [("u", 1, 1), ("u", 3, 3), ("v", 0, 0),
         ("v", 2, 2), el.E[0], el.E[4]])           # 6 repeats, shifted onset
assert len(SC) == 20, "expected 20 scenarios, got %d" % len(SC)
ncc = sum(collision_capable(e) for e in SC)

xb, tb = braking_distance(cfg)
mg = cfg.plant.spacing - cfg.plant.d_min
print("=" * 78)
print("exp1 : %d scenarios x %d trials x %d methods = %d runs"
      % (len(SC), N_TRIAL, len(METHODS), len(SC) * N_TRIAL * len(METHODS)))
print("  w_safety = %.1f   fallback = %s   shared x0 + deep-copied plant = %s"
      % (getattr(cfg, "w_safety", 0.0), FALLBACK, True))
print("  braking distance %.3f m vs nominal margin %.3f m  -> incursion %s"
      % (xb, mg, "unavoidable after onset" if xb > mg else "avoidable"))
print("  collision-capable scenarios: %d of %d  (indices %s)"
      % (ncc, len(SC), [i for i, e in enumerate(SC) if collision_capable(e)]))
print("=" * 78, flush=True)

t0 = time.time()
cr = build_calibration(cfg, Elements, cfg.plant.N, runs=30)
print("calibration : n=%d  rbar_max=%.5f  rho=%.5f  alpha_e=%.5f  (%.1f s)"
      % (cr.n_cal, cr.rbar_max, rho, rc.alpha / len(el), time.time() - t0),
      flush=True)
assert cr.rbar_max > rho, "n_cal too small for this alpha"

# ------------------------------------------------------------------ run
rows = {m: [] for m in METHODS}
det_rows, tag, ccflag = [], [], []
t0 = time.time()
for si, e_f in enumerate(SC):
    c = Cfg()
    if si >= 14:
        c.deg.t_deg = cfg.deg.t_deg + 0.5          # shifted onset
    cc = collision_capable(e_f)
    for tr in range(N_TRIAL):
        pl = Plant(c, el, failing=[e_f],
                   rng=np.random.default_rng(10000 + 131 * si + tr))
        ks, R = risk_trace(pl, cr, rc)
        x0 = pl.x0()                               # drawn ONCE, see (1)
        det_rows.append(detection_stats(ks, R, el.index[e_f], rho,
                                        rc.persistence, pl.k_fail,
                                        pl.dt, len(el)))
        tag.append(e_f[0])
        ccflag.append(cc)
        for m in METHODS:
            H = run(copy.deepcopy(pl), el, c, m, ks, R, rho, x0=x0)   # see (1)
            ss = safety_stats(H)
            rows[m].append(dict(
                rec=recovery_time(H, pl.k_fail, pl.dt),
                min_h=ss["min_h"], nviol=ss["nviol"],
                viol_area=ss["viol_area"], anyviol=ss["nviol"] > 0,
                nslack=ss["nslack"],
                sel_ms=float(np.mean(H["sel_ms"])) if H["sel_ms"] else 0.0,
                nswitch=len(H["switch"]), infeas=H.get("infeas", 0),
                relax=H.get("relax", 0), qpfail=H.get("qpfail", 0)))
    print("  scenario %2d/%2d  %-12s %s (%.0f s elapsed)"
          % (si + 1, len(SC), str(e_f), "*" if cc else " ",
             time.time() - t0), flush=True)

# ------------------------------------------------------------------ report
d = summarize(det_rows)
print("\n" + "=" * 78)
print("Detection (Task A)   n = %d runs" % len(det_rows))
print("=" * 78)
print("  detection rate            : %.1f %%" % (100 * d["detected"][0]))
print("  mean lead time            : %.2f +- %.2f s" % d["lead"])
print("  empirical FAR (per check) : %.4f" % d["far_check"][0])
print("  nominal bound alpha_e     : %.4f   (ratio %.2f; compare with the"
      " calibration band, not with the bound itself)"
      % (rc.alpha / len(el), d["far_check"][0] / (rc.alpha / len(el))))
tag = np.array(tag)
for t in ("u", "v", "e"):
    m = tag == t
    if m.sum():
        dd = summarize([r for r, z in zip(det_rows, m) if z])
        print("    %s-faults (n=%3d): rate %5.1f %%  lead %.2f s"
              % (t, m.sum(), 100 * dd["detected"][0], dd["lead"][0]))

S = {m: summarize(rows[m]) for m in METHODS}
print("\n" + "=" * 78)
print("Table 1 : method comparison   (mean +- sd over %d runs)" % len(rows["prop"]))
print("=" * 78)
hdr = "%-26s %16s %16s %16s" % ("metric", "fixed", "post-failure", "PROPOSED")
print(hdr)
print("-" * len(hdr))
for key, lab, fmt in [("rec", "recovery time [s]", "%.2f+-%.2f"),
                      ("min_h", "min gap margin [m]", "%.3f+-%.3f"),
                      ("nviol", "violation steps", "%.1f+-%.1f"),
                      ("viol_area", "violation area", "%.3f+-%.3f"),
                      ("nslack", "slack activations", "%.1f+-%.1f"),
                      ("nswitch", "structure switches", "%.2f+-%.2f"),
                      ("sel_ms", "selection time [ms]", "%.3f+-%.3f")]:
    v = [(fmt % S[m][key]) if S[m][key][0] == S[m][key][0] else "n/a"
         for m in METHODS]
    print("%-26s %16s %16s %16s" % (lab, *v))

ccf = np.array(ccflag, dtype=bool)
av = {m: np.array([r["anyviol"] for r in rows[m]], dtype=bool)
      for m in METHODS}
print("%-26s %15.0f%% %15.0f%% %15.0f%%"
      % ("violation, all scenarios", 100 * av["fixed"].mean(),
         100 * av["post"].mean(), 100 * av["prop"].mean()))
print("%-26s %15.0f%% %15.0f%% %15.0f%%    <- report this one"
      % ("violation, collision-cap.", 100 * av["fixed"][ccf].mean(),
         100 * av["post"][ccf].mean(), 100 * av["prop"][ccf].mean()))
kc = int(av["prop"][ccf].sum()); nc = int(ccf.sum())
print("%-26s %16s %16s %10d/%d [%.1f,%.1f]"
      % ("  proposed, Wilson 95% CI", "", "", kc, nc,
         *(100 * np.array(wilson(kc, nc)))))

rp = np.array([r["rec"] for r in rows["post"]])
rq = np.array([r["rec"] for r in rows["prop"]])
npair, fa, sl, pw = wilcoxon(rp, rq)
po, pr, pm = mcnemar(av["post"], av["prop"])
po_c, pr_c, pm_c = mcnemar(av["post"][ccf], av["prop"][ccf])
print("\n" + "-" * 78)
print("  Wilcoxon (recovery, paired) : n=%d  prop faster %d / slower %d  p=%.2e"
      % (npair, fa, sl, pw))
print("  McNemar  (violation, all)   : b=post-only=%d  c=prop-only=%d  p=%.2e"
      % (po, pr, pm))
print("  McNemar  (violation, cc)    : b=post-only=%d  c=prop-only=%d  p=%.2e"
      % (po_c, pr_c, pm_c))
if S["post"]["rec"][0] > 0:
    print("  recovery-time change        : %+.1f %%  (positive = proposed faster)"
          % (100 * (S["post"]["rec"][0] - S["prop"]["rec"][0])
             / S["post"]["rec"][0]))
if S["post"]["viol_area"][0] > 0:
    print("  violation-area reduction    : %+.1f %%"
          % (100 * (S["post"]["viol_area"][0] - S["prop"]["viol_area"][0])
             / S["post"]["viol_area"][0]))
print("  QP failures                 : %d"
      % sum(r["qpfail"] for r in rows["prop"]))
print("  infeasible selections       : %d"
      % sum(r["infeas"] for r in rows["prop"]))
print("  relaxed selection calls     : %d"
      % sum(r["relax"] for r in rows["prop"]))

out = {"alpha_e": rc.alpha / len(el), "rho": rho, "n_trial": N_TRIAL,
       "cc": ccf.astype(float)}
for k in det_rows[0]:
    out["det_" + k] = np.array([r[k] for r in det_rows], dtype=float)
for m in METHODS:
    for k in rows[m][0]:
        out["%s_%s" % (m, k)] = np.array([r[k] for r in rows[m]], dtype=float)
if os.path.exists(OUT) and not os.path.exists(BACKUP):
    shutil.copy(OUT, BACKUP)
    print("\nprevious %s kept as %s" % (OUT, BACKUP))
np.savez(OUT, **out)
print("\nsaved %s   total %.0f s  ->  Fig. 3 can now be generated"
      % (OUT, time.time() - t0))
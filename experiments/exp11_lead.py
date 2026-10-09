# -*- coding: utf-8 -*-
"""exp11: oracle-lead sweep (Theorem 1 check), abrupt faults, static safe spacing,
and the theorem policy with structure reconfiguration enabled.

Detector replaced by an oracle: the failing element's risk is set to 1 from
k_fail - lead onward (other elements keep the real conformal trace, so false alarms
are realistic).  'none' = no precursor (abrupt fault); 'real' = actual conformal trace.
Usage: python -X utf8 experiments/exp11_lead.py [--N 4] [--leads real,none,0,0.2,...]
"""
import sys, os, copy, math, argparse, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
from masrecon.metrics import safety_stats, recovery_time

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=4)
ap.add_argument("--leads", default="real,none,0,0.2,0.4,0.5,0.6,0.8,1,2,3")
ap.add_argument("--static", type=float, default=2.03, help="safe spacing delta' [m]")
ap.add_argument("--no-static", action="store_true")
ap.add_argument("--lstar", type=int, default=2)
ap.add_argument("--t-settle", type=float, default=2.0)
ap.add_argument("--tag", default="")
ap.add_argument("--max-runs", type=int, default=0)
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
NAN = float("nan")

TP = dict(w_safety=2.0, policy_T=True, nonretreat=True, hard_barrier=True)
CONDS = [("T_latch_stop",   "prop", dict(TP, freeze_structure=True)),
         ("T_latch_rel10",  "prop", dict(TP, freeze_structure=True, latch_release=10)),
         ("T_latch_reconf", "prop", dict(TP, force_critical=True, fallback_tube="stop")),
         ("C3_force_stop",  "prop", dict(w_safety=2.0, force_critical=True, fallback_tube="stop")),
         ("C0_weight",      "prop", dict(w_safety=2.0))]
POLT = {"T_latch_stop", "T_latch_rel10", "T_latch_reconf"}

def make_cfg(spacing=None, **over):
    c = Cfg()
    if args.N != c.plant.N:
        c.plant.N = args.N          # CHECK: other N-dependent fields in config.py?
    if spacing is not None:
        c.plant.spacing = spacing
    for k, v in over.items():
        setattr(c, k, v)
    return c

def wilson(k, n, z=1.959964):
    if n == 0: return (NAN, NAN)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)

def first_flag(ks, R, rho, P, j):
    cnt = 0
    for i, a in enumerate(np.asarray(R)[:, j] > rho):
        cnt = cnt + 1 if a else 0
        if cnt >= P: return int(ks[i])
    return -1

base = make_cfg(); N = base.plant.N; el = Elements(N); rc = base.risk
rho = 1.0 - rc.alpha / len(el); dt = base.plant.dt
DELTA, DMIN = base.plant.spacing, base.plant.d_min
K0 = int(round(args.t_settle / dt))
cr = build_calibration(base, Elements, N, runs=30)
LEADS = [s.strip() for s in args.leads.split(",") if s.strip()]

runs = []
for si, e in enumerate([("u", i, i) for i in range(1, min(N, 4))]):
    for tr in range(10):
        runs.append(("%s/%d" % (e, tr), e, 10000 + 131 * (si + 1) + tr))
if args.max_runs: runs = runs[:args.max_runs]

rows = []   # dict per (run, cond, lead)
t0 = time.time()
for ri, (lab, e_f, seed) in enumerate(runs):
    pl = Plant(make_cfg(), el, failing=[e_f], rng=np.random.default_rng(seed))
    ks, R = risk_trace(pl, cr, rc); ks = np.asarray(ks); R = np.asarray(R, float)
    x0 = pl.x0(); kf = pl.k_fail; kdeg = pl.k_deg
    jf = e_f[1]; idx = el.index[e_f]
    for lead in LEADS:
        Rs = R.copy()
        if lead != "real":
            Rs[:, idx] = 0.0
            if lead != "none":
                Rs[ks >= kf - int(round(float(lead) / dt)), idx] = 1.0
        kd = first_flag(ks, Rs, rho, rc.persistence, idx)
        lead_act = (kf - kd) * dt if 0 <= kd <= kf else NAN
        for name, method, over in CONDS:
            H = run(copy.deepcopy(pl), el, make_cfg(**over), method, ks, Rs, rho, x0=x0)
            ss = safety_stats(H); rec = recovery_time(H, kf, dt)
            E = -1
            if name in POLT:
                lat = np.asarray(H["latched"])
                E = int(bool(lat[max(kf - args.lstar, 0):kf + 1, jf].all()))
            h = np.asarray(H["h"], float)
            rows.append(dict(run=lab, cond=name, lead=lead, viol=ss["nviol"] > 0,
                             area=ss["viol_area"], min_h=ss["min_h"], rec=rec, E=E,
                             lead_act=lead_act, hr=H.get("hard_relaxed", 0),
                             sh=int((np.asarray(H["shield"]) > 0).sum()),
                             fcost=float(np.abs(h[K0:kdeg] + DMIN - DELTA).mean()),
                             nsw=len(H["switch"])))
    if not args.no_static:
        for name, method, sp in (("B1_post", "post", None), ("S_static", "post", args.static),
                                 ("F_fixed_static", "fixed", args.static)):
            c = make_cfg(spacing=sp, w_safety=2.0)
            p2 = Plant(c, el, failing=[e_f], rng=np.random.default_rng(seed))
            ks2, R2 = risk_trace(p2, cr, rc); x02 = p2.x0()
            H = run(copy.deepcopy(p2), el, c, method, np.asarray(ks2), np.asarray(R2, float),
                    rho, x0=x02)
            ss = safety_stats(H); h = np.asarray(H["h"], float)
            rows.append(dict(run=lab, cond=name, lead="-", viol=ss["nviol"] > 0,
                             area=ss["viol_area"], min_h=ss["min_h"],
                             rec=recovery_time(H, p2.k_fail, dt), E=-1, lead_act=NAN,
                             hr=0, sh=0, nsw=len(H["switch"]),
                             fcost=float(np.abs(h[K0:p2.k_deg] + DMIN - DELTA).mean())))
    print("  %d/%d runs (%.0f s)" % (ri + 1, len(runs), time.time() - t0))

print("\nN=%d  rho=%.4f  L*=%d steps  (viol = any h<0; E = latched on [k_f-L*,k_f])" % (N, rho, args.lstar))
print("%-15s %5s %8s %15s %9s %10s %7s %6s %7s %4s %7s %6s" % (
    "cond", "lead", "viol", "Wilson95", "inE", "viol_inE", "lead_a", "hrel", "shield",
    "nr", "rec", "fcost"))
keys = sorted({(r["cond"], r["lead"]) for r in rows},
              key=lambda x: (x[0], -1e9 if x[1] in ("real", "-") else (-1e8 if x[1] == "none" else float(x[1]))))
for cnd, ld in keys:
    S = [r for r in rows if r["cond"] == cnd and r["lead"] == ld]
    v = np.array([r["viol"] for r in S]); Ev = np.array([r["E"] for r in S])
    lo, hi = wilson(int(v.sum()), len(v)); rec = np.array([r["rec"] for r in S], float)
    la = np.array([r["lead_act"] for r in S], float)
    print("%-15s %5s %3d/%-4d [%5.1f,%5.1f] %9s %10s %7s %6d %7d %4d %7s %6.3f" % (
        cnd, ld, int(v.sum()), len(v), lo, hi,
        ("%d" % (Ev == 1).sum()) if (Ev >= 0).any() else "-",
        ("%d" % (v & (Ev == 1)).sum()) if (Ev >= 0).any() else "-",
        ("%.2f" % np.nanmean(la)) if np.isfinite(la).any() else "-",
        sum(r["hr"] for r in S), sum(r["sh"] for r in S), int((~np.isfinite(rec)).sum()),
        ("%.2f" % np.nanmean(rec)) if np.isfinite(rec).any() else "-",
        np.mean([r["fcost"] for r in S])))
bad = [(r["cond"], r["lead"], r["run"]) for r in rows if r["viol"] and r["E"] == 1]
print("\nH1 COUNTEREXAMPLES (violating runs in E, must be empty): %d %s" % (len(bad), bad[:20]))

fn = "results/exp11_lead%s.npz" % args.tag
np.savez(fn, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
print("saved %s sha256 %s" % (fn, hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]))

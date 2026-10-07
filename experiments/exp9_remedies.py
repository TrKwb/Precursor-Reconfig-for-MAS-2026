# -*- coding: utf-8 -*-
"""exp9 / exp10 : the remedies the theory predicts, the baselines a referee will ask for,
and the theorem policy pi_T (Theorem 1).

Conditions (N = 4):
  C0_weight          Remedy 1 only (w_s=2, nominal-drift fallback)   -- = exp8 at w_s=2
  C1_force           + Remedy 2 (forced critical edges, Prop. 5)
  C2_stopfb          Remedy 1 + immediate-stop fallback (Remark 3(i))
  C3_force_stop      Remedy 2 + immediate-stop fallback
  C4_twotier         C3 + hard h >= 0 (two-tier constraints, Sec. 5.6(b))
  C5_tighten_only    tightening from detection, no pre-emptive reconfiguration
  C6_force_stopalive C3, but the stop tube only while the edge is physically alive
  T_latch_stop       C5 + latch / stop tube / braking-envelope tube / shield,
                     hard barrier, v >= 0                       (Theorem 1 policy)
  T_latch_rel10      T_latch_stop, latch released after 10 unflagged evaluations
  T_latch_soft       T_latch_stop with the soft barrier only (ablation)
  B1_post            post-failure oracle
  B2_always_stop     post-failure + immediate-stop tube for every forward pair
  C0_ws0, C1_force_ws0   the same at w_s = 0

Run sets:
  cc30  : exp8's 30 collision-capable runs
  c1    : exp1's 200 runs.  Scenarios 0-13 at onset +0.0 s (its cc runs ARE cc30),
          scenarios 14-19 = repeats with other seeds at onset +0.5 s
  onset : cc30 at several onset shifts (--shifts, default 0,2,4,5; negatives allowed,
          write them as --shifts=-3,-2,0)

Time windows (identical for every condition of a run, fixed by R only):
  pure-healthy [t_settle, k_deg)  healthy [0, k_det)  warn [k_det, k_fail)  post [k_fail, T)
  k_det = first persistent flag of the FAILING element (k_fail if not detected before
  failure).

False alarms (all from the risk trace, condition-independent):
  other-element flag  : an element other than the failing one flagged before k_fail
  pre-onset FA        : an element other than the failing one flagged before k_deg
                        (pure-healthy cost is split 3 ways: none / failing element only / other)
  noise-matched       : the fault-free TWIN (same seed, no failing element) flags the same
                        element no later -> the flag is noise, not caused by the fault
  fault-induced       : flagged in the faulty run earlier than in the twin (or only there)

Theorem 1 check (policy_T conditions): E = the failing actuator is latched throughout
[k_fail - L*, k_fail] (L* = --lstar steps, default 2 = 0.10 s).  Every violating run must
lie outside E; a violating run in E is an H1 COUNTEREXAMPLE.

Outputs (suffix --tag, default '' ; use e.g. --tag _v10 to keep older result files):
  results/exp9_<set><tag>.npz                 per-run metrics of every condition
  results/exp9_<set><tag>_struct_runs.pkl     histories of detected-but-violated cc runs
                                              (keys 'cond|label|shift', as before)

Usage:
  python experiments/exp9_remedies.py --set cc30
  python experiments/exp9_remedies.py --set onset --shifts=-3,-2,-1,0,2,4,5
  python experiments/exp9_remedies.py --set c1 --only C5_tighten_only,T_latch_stop --tag _v10
  python experiments/exp9_remedies.py --set cc30 --keys      # print H keys, exit
"""
import sys, os, copy, math, csv, hashlib, argparse, time, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
from masrecon.metrics import safety_stats, recovery_time, detection_stats

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["cc30", "c1", "onset"], default="cc30")
ap.add_argument("--only", default="", help="comma-separated condition names")
ap.add_argument("--shifts", default="", help="onset shifts [s], e.g. --shifts=-3,-2,0,2,4")
ap.add_argument("--dump",
                default="C0_weight,C3_force_stop,C5_tighten_only,C6_force_stopalive,T_latch_stop",
                help="conditions whose detected-but-violated cc runs are dumped")
ap.add_argument("--t-settle", type=float, default=2.0,
                help="[s] start of the pure-healthy window [t_settle, t_deg)")
ap.add_argument("--no-twin", action="store_true", help="skip the fault-free twin")
ap.add_argument("--lstar", type=int, default=2, help="required lead L* [steps] (Theorem 1)")
ap.add_argument("--tag", default="", help="suffix of the output file names")
ap.add_argument("--keys", action="store_true",
                help="print the keys of H and det for the first run, then exit")
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
NAN = float("nan")

CONDITIONS = [
    ("C0_weight",          "prop", dict(w_safety=2.0)),
    ("C1_force",           "prop", dict(w_safety=2.0, force_critical=True)),
    ("C2_stopfb",          "prop", dict(w_safety=2.0, fallback_tube="stop")),
    ("C3_force_stop",      "prop", dict(w_safety=2.0, force_critical=True,
                                        fallback_tube="stop")),
    ("C4_twotier",         "prop", dict(w_safety=2.0, force_critical=True,
                                        fallback_tube="stop", hard_barrier=True)),
    ("C5_tighten_only",    "prop", dict(w_safety=2.0, freeze_structure=True)),
    ("C6_force_stopalive", "prop", dict(w_safety=2.0, force_critical=True,
                                        fallback_tube="stop_alive")),
    ("T_latch_stop",       "prop", dict(w_safety=2.0, freeze_structure=True, policy_T=True,
                                        nonretreat=True, hard_barrier=True)),
    ("T_latch_rel10",      "prop", dict(w_safety=2.0, freeze_structure=True, policy_T=True,
                                        nonretreat=True, hard_barrier=True,
                                        latch_release=10)),
    ("T_latch_soft",       "prop", dict(w_safety=2.0, freeze_structure=True, policy_T=True,
                                        nonretreat=True)),
    ("B1_post",            "post", dict(w_safety=2.0)),
    ("B2_always_stop",     "post", dict(w_safety=2.0, always_stop=True)),
    ("C0_ws0",             "prop", dict(w_safety=0.0)),
    ("C1_force_ws0",       "prop", dict(w_safety=0.0, force_critical=True)),
]
# wiring: name -> (reference, difference_required)
REFERENCE = {"C1_force": ("C0_weight", False),
             "C2_stopfb": ("C0_weight", True),
             "C3_force_stop": ("C0_weight", True),
             "C4_twotier": ("C3_force_stop", True),
             "C5_tighten_only": ("C0_weight", True),
             "C6_force_stopalive": ("C3_force_stop", False),   # equal on u-only sets
             "T_latch_stop": ("C5_tighten_only", True),
             "T_latch_rel10": ("T_latch_stop", False),
             "T_latch_soft": ("T_latch_stop", False),
             "B2_always_stop": ("B1_post", True),
             "C1_force_ws0": ("C0_ws0", True)}
# paired tests (candidate, reference)
MCNEMAR = [("C1_force", "C0_weight"), ("C2_stopfb", "C0_weight"),
           ("C3_force_stop", "C0_weight"), ("C4_twotier", "C3_force_stop"),
           ("C5_tighten_only", "C0_weight"), ("C5_tighten_only", "C3_force_stop"),
           ("C6_force_stopalive", "C0_weight"), ("C6_force_stopalive", "C3_force_stop"),
           ("T_latch_stop", "C5_tighten_only"), ("T_latch_stop", "C0_weight"),
           ("T_latch_stop", "B1_post"), ("T_latch_rel10", "T_latch_stop"),
           ("T_latch_soft", "T_latch_stop"),
           ("B2_always_stop", "B1_post"), ("C1_force_ws0", "C0_ws0")]

if args.set == "onset" and not args.only:
    CONDITIONS = [c for c in CONDITIONS if c[0] in (
        "C0_weight", "C3_force_stop", "C5_tighten_only", "C6_force_stopalive",
        "T_latch_stop", "B1_post")]
if args.only:
    keep = [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = set(keep) - {c[0] for c in CONDITIONS}
    assert not unknown, "unknown condition(s): %s" % sorted(unknown)
    CONDITIONS = [c for c in CONDITIONS if c[0] in keep]
CNAMES = [c[0] for c in CONDITIONS]
POLT = {c[0] for c in CONDITIONS if c[2].get("policy_T")}
DUMP = set(args.dump.split(",")) if args.dump else set()

FLAGS = ("force_critical", "fallback_tube", "always_stop", "hard_barrier",
         "freeze_structure")          # pi_T flags are read with defaults by the runner
_c = Cfg()
missing = [f for f in FLAGS if not hasattr(_c, f)]
assert not missing, "add these defaults to Cfg first: %s" % missing


def make_cfg(t_deg_shift=0.0, **over):
    c = Cfg()
    c.deg.t_deg = c.deg.t_deg + t_deg_shift
    for k, v in over.items():
        setattr(c, k, v)
    return c


def wilson(k, n, z=1.959964):
    if n == 0:
        return (NAN,) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(a, b):
    """a, b bool (True = violation).  b10 = a only (b improves), b01 = b only."""
    b10 = int(np.sum(a & ~b)); b01 = int(np.sum(~a & b)); n = b10 + b01
    if n == 0:
        return b10, b01, 1.0
    k = min(b10, b01)
    return b10, b01, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def segmean(a):
    a = np.asarray(a, float)
    if a.size == 0 or not np.isfinite(a).any():
        return NAN
    return float(np.nanmean(a))


def nm(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else NAN


def fx(x, w=7, p=3):
    return ("%*.*f" % (w, p, x)) if np.isfinite(x) else "%*s" % (w, "-")


def first_flag(ks, R, rho, P, j):
    """First plant step at which element j has been above rho for P consecutive
    risk evaluations; -1 if never.  R must be (len(ks), n_elements)."""
    r = np.asarray(R, float)
    assert r.ndim == 2 and r.shape[0] == len(ks), \
        "R has shape %s, expected (len(ks)=%d, n_el)" % (r.shape, len(ks))
    cnt = 0
    for i, a in enumerate(r[:, j] > rho):
        cnt = cnt + 1 if a else 0
        if cnt >= P:
            return int(ks[i])
    return -1


def seg_stats(H, kd, kf, k0, kdeg):
    """ferr in the four windows, mean speed in the pure-healthy and healthy windows,
    err in the post window, first violation and the first violated pair."""
    out = dict(ferr_0=NAN, ferr_h=NAN, ferr_w=NAN, ferr_p=NAN, v_0=NAN, v_h=NAN,
               err_p=NAN, k_viol=-1, pair=-1)
    if "h" not in H:
        return out
    h = np.asarray(H["h"], float)
    if h.ndim == 1:
        h = h[:, None]
    T = h.shape[0]
    kf_ = T if (kf is None or kf < 0) else int(min(kf, T))
    kd_ = kf_ if (kd is None or kd < 0) else int(min(kd, kf_))
    k0_, kg_ = int(min(max(k0, 0), T)), int(min(max(kdeg, 0), T))
    e = np.abs(h - gap0)
    out.update(ferr_0=segmean(e[k0_:kg_]), ferr_h=segmean(e[:kd_]),
               ferr_w=segmean(e[kd_:kf_]), ferr_p=segmean(e[kf_:]))
    if "v" in H:
        v = np.asarray(H["v"], float)
        out["v_0"] = segmean(v[k0_:kg_]); out["v_h"] = segmean(v[:kd_])
    if "err" in H:
        out["err_p"] = segmean(np.asarray(H["err"], float)[kf_:])
    hmin = np.where(np.isfinite(h), h, np.inf).min(axis=1)
    bad = np.where(hmin < 0)[0]
    if bad.size:
        out["k_viol"] = int(bad[0]); out["pair"] = int(np.argmin(h[bad[0]]))
    return out


# ---------------------------------------------------------------- run sets
base = Cfg(); el = Elements(base.plant.N); rc = base.risk
rho = 1.0 - rc.alpha / len(el)
gap0 = base.plant.spacing - base.plant.d_min
ENAME = {v: k for k, v in el.index.items()}
K0 = int(round(args.t_settle / base.plant.dt))

runs = []   # (label, scenario, trial, seed, onset shift, collision-capable, scen idx)
if args.set in ("cc30", "onset"):
    if args.set == "cc30":
        shifts = [0.0]
    elif args.shifts:
        shifts = [float(s) for s in args.shifts.split(",")]
    else:
        shifts = [0.0, 2.0, 4.0, 5.0]
    for sh in shifts:
        for si, e in enumerate([("u", i, i) for i in range(1, base.plant.N)]):
            for tr in range(10):
                runs.append(("%s/%d" % (e, tr), e, tr,
                             10000 + 131 * (si + 1) + tr, sh, True, si))
else:
    SC = ([("u", i, i) for i in range(4)] + [("v", i, i) for i in range(4)]
          + list(el.E) + [("u", 1, 1), ("u", 3, 3), ("v", 0, 0), ("v", 2, 2),
                          el.E[0], el.E[4]])
    for si, e in enumerate(SC):
        for tr in range(10):
            runs.append(("%s/%d" % (e, tr), e, tr, 10000 + 131 * si + tr,
                         0.5 if si >= 14 else 0.0, e[0] == "u" and e[1] > 0, si))

print("=" * 96)
print("exp9 [%s] : %d runs x %d conditions   (tag '%s', L* = %d steps, twin %s)"
      % (args.set, len(runs), len(CONDITIONS), args.tag, args.lstar,
         "off" if args.no_twin else "on"))
print("=" * 96)
cr = build_calibration(base, Elements, base.plant.N, runs=30)

res = {name: [] for name in CNAMES}
meta = []
keepH = {}
n_det_mismatch = n_viol_mismatch = 0
tw_pre_mismatch = tw_ks_mismatch = tw_post_dev_runs = 0
tw_post_dev_max = 0.0
crit_values = set()
t0 = time.time()
for i, (lab, e_f, tr, seed, sh, cc, si) in enumerate(runs):
    c_plant = make_cfg(sh)
    assert c_plant.deg.t_deg > 0, "onset shift %+.1f makes t_deg <= 0" % sh
    pl = Plant(c_plant, el, failing=[e_f], rng=np.random.default_rng(seed))
    ks, R = risk_trace(pl, cr, rc)
    ks = np.asarray(ks); R = np.asarray(R, float)
    x0 = pl.x0()
    kf = pl.k_fail
    kdeg = pl.k_deg
    kf_cut = kf if kf is not None else 10 ** 9
    idx_f = el.index[e_f]
    det = detection_stats(ks, R, idx_f, rho, rc.persistence, kf, pl.dt, len(el))

    kflag = [first_flag(ks, R, rho, rc.persistence, j) for j in range(len(el))]
    k_true = kflag[idx_f]
    det_mine = (kf is not None) and 0 <= k_true <= kf
    if det_mine != bool(det["detected"]):
        n_det_mismatch += 1
    kd = k_true if det_mine else kf

    # ---- fault-free twin (same seed, no failing element) ----------------
    kflag_t = [None] * len(el)
    if not args.no_twin:
        pt = Plant(c_plant, el, failing=[], rng=np.random.default_rng(seed))
        ks_t, R_t = risk_trace(pt, cr, rc)
        ks_t = np.asarray(ks_t); R_t = np.asarray(R_t, float)
        if ks_t.shape != ks.shape or not np.array_equal(ks_t, ks):
            tw_ks_mismatch += 1
        else:
            kflag_t = [first_flag(ks, R_t, rho, rc.persistence, j) for j in range(len(el))]
            pre = ks < kdeg
            if pre.any() and not np.array_equal(R[pre], R_t[pre]):
                tw_pre_mismatch += 1
            others = [j for j in range(len(el)) if j != idx_f]
            dev = (float(np.abs(R[~pre][:, others] - R_t[~pre][:, others]).max())
                   if (~pre).any() else 0.0)
            tw_post_dev_max = max(tw_post_dev_max, dev)
            tw_post_dev_runs += int(dev > 0)

    def origin(j):
        """'n' noise-matched, 'f' fault-induced, '?' no twin."""
        if kflag_t[j] is None:
            return "?"
        return "n" if (0 <= kflag_t[j] <= kflag[j]) else "f"

    fa = [j for j in range(len(el)) if j != idx_f and 0 <= kflag[j] < kf_cut]
    fa_pre_other = [j for j in range(len(el)) if j != idx_f and 0 <= kflag[j] < kdeg]
    fa_pre_self = 0 <= kflag[idx_f] < kdeg
    fa_noise = [j for j in fa if origin(j) == "n"]
    fa_fault = [j for j in fa if origin(j) == "f"]
    jf = e_f[1] if (e_f[0] == "u" and e_f[1] >= 1) else None
    ci = el.index.get(("e", jf - 1, jf)) if jf is not None else None
    cflag = ci is not None and 0 <= kflag[ci] < kf_cut
    meta.append(dict(label=lab, scenario=str(e_f), trial=tr, shift=sh, cc=cc,
                     si=si, detected=bool(det["detected"]), typ=e_f[0],
                     jf=-1 if jf is None else jf,
                     k_fail=-1 if kf is None else int(kf), k_deg=int(kdeg),
                     k_det=k_true if det_mine else -1,
                     lead=(kf - k_true) * pl.dt if det_mine else NAN,
                     fa=len(fa) > 0, fa_pre=len(fa_pre_other) > 0,
                     fa_pre_self=bool(fa_pre_self),
                     fa_noise=len(fa_noise) > 0, fa_fault=len(fa_fault) > 0,
                     tw_any=any(kflag_t[j] is not None and 0 <= kflag_t[j] < kf_cut
                                for j in range(len(el))),
                     cflag=bool(cflag), cflag_o=origin(ci) if cflag else "",
                     fa_names=";".join(str(ENAME[j]) for j in fa_pre_other),
                     fa_noise_names=";".join(str(ENAME[j]) for j in fa_noise
                                             if j not in fa_pre_other),
                     fa_fault_names=";".join(str(ENAME[j]) for j in fa_fault),
                     t_deg=c_plant.deg.t_deg,
                     t_fail=NAN if kf is None else kf * pl.dt, T=pl.K * pl.dt))

    for name, method, over in CONDITIONS:
        cfg = make_cfg(sh, **over)
        H = run(copy.deepcopy(pl), el, cfg, method, ks, R, rho, x0=x0)
        if args.keys:
            print("det keys:", sorted(det.keys()))
            print("H keys (condition %s):" % name)
            for k_ in sorted(H.keys()):
                v_ = H[k_]
                try:
                    shp = np.asarray(v_).shape
                except Exception:
                    shp = "?"
                print("   %-20s %-12s %s" % (k_, type(v_).__name__, shp))
            print("R shape:", R.shape, " len(ks):", len(ks), " k_fail:", kf, " dt:", pl.dt)
            sys.exit(0)
        ss = safety_stats(H)
        r = recovery_time(H, kf, pl.dt)
        sg = seg_stats(H, kd, kf, K0, kdeg)
        if (sg["k_viol"] >= 0) != (ss["nviol"] > 0):
            n_viol_mismatch += 1
        crit = H.get("crit_kept", -1)
        crit_values.add(crit if np.isscalar(crit) else str(crit))
        cw = -1                                   # critical edge kept over the whole warn window
        if jf is not None and det_mine and "crit_in" in H:
            ci_arr = np.asarray(H["crit_in"])
            seg = ci_arr[kd:kf] if kf is not None else ci_arr[kd:]
            cw = int(seg.size == 0 or bool((seg == 1).all()))
        E = -1; shield = 0                        # Theorem 1 event (policy_T only)
        if name in POLT and jf is not None and kf is not None and "latched" in H:
            lat = np.asarray(H["latched"])
            if lat.ndim == 2 and lat.shape[0] > kf:
                E = int(bool(lat[max(kf - args.lstar, 0):kf + 1, jf].all()))
            shield = int((np.asarray(H.get("shield", [0])) > 0).sum())
        row = dict(viol=ss["nviol"] > 0, area=ss["viol_area"],
                   min_h=ss["min_h"], nslack=ss["nslack"],
                   rec=r, nsw=len(H["switch"]),
                   qpfail=H.get("qpfail", 0), crit=crit, crit_w=cw,
                   hrelax=H.get("hard_relaxed", 0), nrrelax=H.get("nonretreat_relaxed", 0),
                   E=E, shield=shield, **sg)
        res[name].append(row)
        if name in DUMP and row["viol"] and cc and det["detected"]:
            keepH[(name, i)] = H
    if (i + 1) % 50 == 0:
        print("  %d/%d runs  (%.0f s)" % (i + 1, len(runs), time.time() - t0))
print("  done in %.0f s" % (time.time() - t0))

# ---------------------------------------------- diagnostics
print("  diag: own first_flag vs detection_stats 'detected' differ in %d / %d runs"
      % (n_det_mismatch, len(runs)))
print("  diag: own first violation (h<0) vs safety_stats nviol>0 differ in %d / %d"
      " run-conditions" % (n_viol_mismatch, len(runs) * len(CONDITIONS)))
print("  diag: crit_kept values returned by runner: %s" % sorted(map(str, crit_values)))
if not args.no_twin:
    print("  diag: twin evaluation times (ks) differ in %d / %d runs" % (tw_ks_mismatch, len(runs)))
    print("  diag: twin vs faulty pre-onset risk mismatch in %d / %d runs (must be 0)"
          % (tw_pre_mismatch, len(runs)))
    print("  diag: post-onset |R - R_twin| on non-failing elements: max %.3g, deviating runs"
          " %d / %d" % (tw_post_dev_max, tw_post_dev_runs, len(runs)))
    print("  diag: trajectory-level FAR (fault-free twin, any flag before k_fail): %d / %d runs"
          % (sum(m["tw_any"] for m in meta), len(meta)))

# ---------------------------------------------- check 1: reproduce exp8
if args.set == "cc30":
    if not os.path.exists("results/exp8_wsafety.csv"):
        print("  !! WARNING: results/exp8_wsafety.csv not found -- the exp8 "
              "reproduction check was SKIPPED")
    else:
        ref = {}
        with open("results/exp8_wsafety.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ref[(float(row["w_safety"]), row["scenario"], int(row["trial"]))] = \
                    int(row["violation"])
        for name, ws in (("C0_weight", 2.0), ("C0_ws0", 0.0)):
            if name not in res:
                continue
            bad = [m["label"] for m, x in zip(meta, res[name])
                   if ref.get((ws, m["scenario"], m["trial"])) != int(x["viol"])]
            assert not bad, "%s does not reproduce exp8 at w_s=%g: %s" % (name, ws, bad)
            print("  CHECK passed: %s reproduces exp8 (w_s=%g) run for run" % (name, ws))

# ---------------------------------------------- check 2: B1 keeps the critical edge
if "B1_post" in res:
    cv = [x["crit"] for m, x in zip(meta, res["B1_post"]) if m["cc"]]
    nok = sum(1 for c in cv if c == 1)
    print("  CHECK %s: B1_post keeps the critical edge until failure in %d/%d cc u-runs"
          % ("passed" if nok == len(cv) else "FAILED", nok, len(cv)))

# ---------------------------------------------- check 3: flags are wired
def sig(x):
    return (x["viol"], x["nslack"], x["nsw"], round(x["min_h"], 4),
            round(x["ferr_h"], 6) if np.isfinite(x["ferr_h"]) else None)
for name, (refname, required) in REFERENCE.items():
    if name in res and refname in res:
        nd = sum(sig(a) != sig(b) for a, b in zip(res[name], res[refname]))
        note = ""
        if nd == 0:
            note = ("   !! flag NOT wired (a difference is required)" if required
                    else "   (allowed: no difference is required here)")
        print("  wiring: %-18s differs from %-16s in %3d runs%s"
              % (name, refname, nd, note))

# ---------------------------------------------------------------- report
def col(name, k):
    return np.array([x[k] for x in res[name]], dtype=float)
def mcol(k, dtype=float):
    return np.array([m[k] for m in meta], dtype=dtype)
ccm = mcol("cc", bool); detm = mcol("detected", bool); fam = mcol("fa", bool)
fapre = mcol("fa_pre", bool); faself = mcol("fa_pre_self", bool)
fanoise = mcol("fa_noise", bool); fafault = mcol("fa_fault", bool)
cflagm = mcol("cflag", bool); cfo = np.array([m["cflag_o"] for m in meta])
typ = np.array([m["typ"] for m in meta]); shm = mcol("shift"); sim = mcol("si", int)
leadm = mcol("lead"); tdeg = mcol("t_deg"); tfail = mcol("t_fail"); Tm = mcol("T")

if args.set == "c1":
    GROUPS = [("c1 main : scenarios 0-13, onset +0.0 s  (cc runs = cc30)", sim < 14),
              ("c1 repeat : scenarios 14-19, other seeds, onset +0.5 s", sim >= 14),
              ("c1 pooled : all %d runs" % len(meta), np.ones(len(meta), bool))]
elif args.set == "onset":
    GROUPS = [("onset shift %+.1f s" % s, shm == s) for s in sorted(set(shm))]
else:
    GROUPS = [("cc30", np.ones(len(meta), bool))]

HDR = ("%-18s %9s %15s %4s %6s %7s %7s %4s %7s %7s %7s %7s %6s %6s"
       % ("condition", "viol(cc)", "Wilson 95%", "miss", "struct", "crit",
          "rec(cc)", "nr", "ferr_0", "ferr_h", "ferr_w", "ferr_p", "v_0", "qpfail"))

for title, sel in GROUPS:
    m_cc = ccm & sel
    d_cc = m_cc & detm
    print("\n" + "-" * 96)
    print(title)
    print("  detector: t_deg %s s, t_fail %s s | detected %d/%d cc runs | "
          "lead (detected cc) mean %s s, min %s s"
          % (",".join("%.1f" % x for x in sorted(set(tdeg[sel]))),
             ",".join("%.1f" % x for x in sorted(set(tfail[sel][np.isfinite(tfail[sel])]))),
             int(d_cc.sum()), int(m_cc.sum()),
             fx(nm(leadm[d_cc]), 0, 2),
             fx(np.nanmin(leadm[d_cc]) if d_cc.any() else NAN, 0, 2)))
    print("  other-element flags before k_fail: any %d | pre-onset (true FA) %d | noise-matched %d"
          " | fault-induced %d  ||  critical edge flagged in %d cc runs (noise %d, fault %d)"
          "   (of %d runs)"
          % (int((fam & sel).sum()), int((fapre & sel).sum()), int((fanoise & sel).sum()),
             int((fafault & sel).sum()), int((cflagm & m_cc).sum()),
             int((cflagm & m_cc & (cfo == "n")).sum()),
             int((cflagm & m_cc & (cfo == "f")).sum()), int(sel.sum())))
    print("  windows over cc runs: ferr_0 [t_settle,t_deg)  ferr_h [0,k_det)  "
          "ferr_w [k_det,k_fail)  ferr_p [k_fail,T)   nr = not recovered")
    print(HDR)
    for name in CNAMES:
        V = col(name, "viol").astype(bool)
        n = int(m_cc.sum()); k = int((V & m_cc).sum())
        lo, hi = wilson(k, n)
        miss = int((V & m_cc & ~detm).sum()); struct = k - miss
        cv = col(name, "crit")[d_cc]; cv = cv[cv >= 0]
        crit_s = "%d/%d" % (int((cv == 1).sum()), cv.size) if cv.size else "n/a"
        rec = col(name, "rec")[m_cc]
        print("%-18s %4d/%-4d [%5.1f,%5.1f] %4d %6d %7s %s %4d %s %s %s %s %s %6d"
              % (name, k, n, lo, hi, miss, struct, crit_s,
                 fx(nm(rec), 7, 2), int(np.sum(~np.isfinite(rec))),
                 fx(nm(col(name, "ferr_0")[m_cc])), fx(nm(col(name, "ferr_h")[m_cc])),
                 fx(nm(col(name, "ferr_w")[m_cc])), fx(nm(col(name, "ferr_p")[m_cc])),
                 fx(nm(col(name, "v_0")[m_cc]), 6, 3), int(col(name, "qpfail")[sel].sum())))
        cw = col(name, "crit_w")[d_cc]; cw = cw[cw >= 0]
        sv = V & d_cc
        print("%18s crit kept whole warn window %s   | struct viol: crit flagged %d "
              "(noise %d, fault %d), not flagged %d"
              % ("", "%d/%d" % (int((cw == 1).sum()), cw.size) if cw.size else "n/a",
                 int((sv & cflagm).sum()), int((sv & cflagm & (cfo == "n")).sum()),
                 int((sv & cflagm & (cfo == "f")).sum()), int((sv & ~cflagm).sum())))
        if name in POLT:
            Ev = col(name, "E")[m_cc]; Vc = V[m_cc]
            inE, outE = Ev == 1, Ev == 0
            print("%18s theorem: E (latched on [k_f-L*,k_f]) %d/%d | violations in E %d  <- H1 "
                  "counterexamples | violations outside E %d of %d | hard_relaxed calls %d | "
                  "nonretreat_relaxed %d | shield steps %d"
                  % ("", int(inE.sum()), int(m_cc.sum()), int((Vc & inE).sum()),
                     int((Vc & outE).sum()), int(outE.sum()),
                     int(col(name, "hrelax")[sel].sum()), int(col(name, "nrrelax")[sel].sum()),
                     int(col(name, "shield")[sel].sum())))
    for cand, refname in MCNEMAR:
        if cand in res and refname in res:
            a = col(refname, "viol").astype(bool) & m_cc
            b = col(cand, "viol").astype(bool) & m_cc
            b10, b01, p = mcnemar(a, b)
            print("  McNemar %-18s vs %-16s (cc): improved %d, degraded %d, p=%.2e"
                  % (cand, refname, b10, b01, p))
    for nm_ in CNAMES:
        if nm_ == "C4_twotier" or (nm_ in POLT and nm_ != "T_latch_soft"):
            print("  %s hard barrier relaxed to soft in %d QP calls (total)"
                  % (nm_, int(col(nm_, "hrelax")[sel].sum())))

    print("  pure-healthy window [t_settle, t_deg) (all runs): ferr_0 / v_0, split by pre-onset FA")
    splits = (("no FA", sel & ~fapre & ~faself), ("self only", sel & ~fapre & faself),
              ("other FA", sel & fapre))
    for name in CNAMES:
        out = ["%s n=%3d ferr_0 %s v_0 %s" % (l_, int(m_.sum()), fx(nm(col(name, "ferr_0")[m_])),
                                              fx(nm(col(name, "v_0")[m_]), 6, 3))
               for l_, m_ in splits]
        print("    %-18s %s" % (name, "  |  ".join(out)))
    print("  pre-detection window [0, k_det) (all runs): ferr_h / v_h, split by any "
          "other-element flag before k_fail")
    for name in CNAMES:
        out = ["%s n=%3d ferr_h %s v_h %s" % (l_, int(m_.sum()), fx(nm(col(name, "ferr_h")[m_])),
                                              fx(nm(col(name, "v_h")[m_]), 6, 3))
               for l_, m_ in (("no flag", sel & ~fam), ("flag", sel & fam))]
        print("    %-18s %s" % (name, "  |  ".join(out)))

    if args.set == "c1":
        print("  by fault type:  n | rec mean (finite) | rec median (nr censored at T) | nr | "
              "ferr_p | err_p | ferr_0")
        for t in ("u", "v", "e"):
            mt = (typ == t) & sel
            if not mt.any():
                continue
            for name in CNAMES:
                rec = col(name, "rec")[mt]
                cens = np.where(np.isfinite(rec), rec, Tm[mt] - tfail[mt])
                print("    %s  %-18s n=%3d  rec %s  med %s  nr %3d  ferr_p %s  err_p %s  ferr_0 %s"
                      % (t, name, int(mt.sum()), fx(nm(rec), 6, 2),
                         fx(float(np.median(cens)), 6, 2),
                         int(np.sum(~np.isfinite(rec))), fx(nm(col(name, "ferr_p")[mt])),
                         fx(nm(col(name, "err_p")[mt])), fx(nm(col(name, "ferr_0")[mt]))))

# ---------------------------------------------- detected-but-violated dump
dump_names = [n for n in CNAMES if n in DUMP]
if dump_names:
    print("\n" + "-" * 96)
    print("detected-but-violated cc runs   pair = first violated gap (p, p+1); jf = failing agent;"
          "  cfl = critical edge flagged (n = noise, f = fault)")
    print("%-18s %-16s %6s %3s %6s %6s %6s %7s %4s %3s %7s %4s"
          % ("condition", "run", "shift", "jf", "pair", "k_det", "k_fail", "lead s", "crit",
             "cfl", "viol-f s", "nsw"))
    for name in dump_names:
        for i, (m, x) in enumerate(zip(meta, res[name])):
            if not (x["viol"] and m["cc"] and m["detected"]):
                continue
            dtv = ((x["k_viol"] - m["k_fail"]) * (m["t_fail"] / m["k_fail"])
                   if x["k_viol"] >= 0 and m["k_fail"] > 0 else NAN)
            sw = keepH.get((name, i), {}).get("switch", [])
            pr = "(%d,%d)" % (x["pair"], x["pair"] + 1) if x["pair"] >= 0 else "-"
            print("%-18s %-16s %+6.1f %3d %6s %6d %6d %s %4s %3s %s %4d"
                  % (name, m["label"], m["shift"], m["jf"], pr, m["k_det"], m["k_fail"],
                     fx(m["lead"], 7, 2), str(int(x["crit"])), m["cflag_o"] or "-",
                     fx(dtv, 7, 2), int(x["nsw"])))
            print("%18s flags: pre-onset[%s] noise[%s] fault[%s]  switches %s%s"
                  % ("", m["fa_names"], m["fa_noise_names"], m["fa_fault_names"],
                     [round(float(s), 2) for s in list(sw)[:6]],
                     ("  E=%d" % x["E"]) if name in POLT else ""))
    pk = "results/exp9_%s%s_struct_runs.pkl" % (args.set, args.tag)
    try:
        with open(pk, "wb") as f:
            pickle.dump({"%s|%s|%+.1f" % (n, meta[i]["label"], meta[i]["shift"]): H
                         for (n, i), H in keepH.items()}, f)
        print("  saved %d histories to %s" % (len(keepH), pk))
    except Exception as ex:
        print("  !! could not pickle histories: %s" % ex)

# ---------------------------------------------------------------- save
out = {"cc": ccm, "detected": detm, "shift": shm, "si": sim, "typ": typ,
       "fa": fam, "fa_pre": fapre, "fa_pre_self": faself, "fa_noise": fanoise,
       "fa_fault": fafault, "cflag": cflagm, "cflag_o": cfo,
       "lead": leadm, "t_deg": tdeg, "t_fail": tfail,
       "k_fail": mcol("k_fail", int), "k_det": mcol("k_det", int),
       "k_deg": mcol("k_deg", int), "labels": np.array([m["label"] for m in meta])}
for name in CNAMES:
    for k in ("viol", "area", "min_h", "nslack", "rec", "nsw", "qpfail", "crit", "crit_w",
              "hrelax", "nrrelax", "E", "shield", "ferr_0", "ferr_h", "ferr_w", "ferr_p",
              "v_0", "v_h", "err_p", "k_viol", "pair"):
        out["%s__%s" % (name, k)] = col(name, k)
fn = "results/exp9_%s%s.npz" % (args.set, args.tag)
np.savez(fn, **out)
print("\nsaved %s  sha256 %s" % (fn, hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]))
# -*- coding: utf-8 -*-
"""exp9 : the remedies the theory predicts, and the baselines a referee will ask for.

Conditions (N = 4):
  C0_weight       Remedy 1 only (w_s=2, nominal-drift fallback)   -- = exp8 at w_s=2
  C1_force        + Remedy 2 (forced critical edges, Prop. 5)
  C2_stopfb       Remedy 1 + immediate-stop fallback (Remark 3(i))
  C3_force_stop   Remedy 2 + immediate-stop fallback
  C4_twotier      C3 + hard h >= 0 (two-tier constraints, Sec. 5.6(b))
  C5_tighten_only tightening from detection, no pre-emptive reconfiguration
  B1_post         post-failure oracle
  B2_always_stop  post-failure + immediate-stop tube for every forward pair
  C0_ws0, C1_force_ws0   the same at w_s = 0

Run sets:
  cc30  : exp8's 30 collision-capable runs
  c1    : exp1's 200 runs.  Scenarios 0-13 at onset +0.0 s (its cc runs ARE cc30),
          scenarios 14-19 = repeats with other seeds at onset +0.5 s
  onset : cc30 at several onset shifts (--shifts, default 0,2,4,5; negatives allowed)

Time windows (identical for every condition of a run, fixed by R only):
  healthy [0, k_det)   warn [k_det, k_fail)   post [k_fail, T)
  k_det = first persistent flag of the FAILING element (k_fail if not detected
  before failure).  False alarm = another element flagged before k_fail.

Checks before saving:
  * C0_weight / C0_ws0 on cc30 reproduce exp8 run for run;
  * each flag is compared with its reference;
  * own detection (first_flag) vs detection_stats, own violation time vs
    safety_stats -- mismatches are COUNTED and printed, not hidden.

Usage:
  python experiments/exp9_remedies.py --set cc30
  python experiments/exp9_remedies.py --set onset --shifts -3,-2,-1,0,2,4
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
ap.add_argument("--shifts", default="", help="onset shifts [s], e.g. -3,-2,0,2,4")
ap.add_argument("--dump", default="C0_weight,C3_force_stop",
                help="conditions whose detected-but-violated cc runs are dumped")
ap.add_argument("--keys", action="store_true",
                help="print the keys of H and det for the first run, then exit")
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
NAN = float("nan")

CONDITIONS = [
    ("C0_weight",       "prop", dict(w_safety=2.0)),
    ("C1_force",        "prop", dict(w_safety=2.0, force_critical=True)),
    ("C2_stopfb",       "prop", dict(w_safety=2.0, fallback_tube="stop")),
    ("C3_force_stop",   "prop", dict(w_safety=2.0, force_critical=True,
                                     fallback_tube="stop")),
    ("C4_twotier",      "prop", dict(w_safety=2.0, force_critical=True,
                                     fallback_tube="stop", hard_barrier=True)),
    ("C5_tighten_only", "prop", dict(w_safety=2.0, freeze_structure=True)),
    ("B1_post",         "post", dict(w_safety=2.0)),
    ("B2_always_stop",  "post", dict(w_safety=2.0, always_stop=True)),
    ("C0_ws0",          "prop", dict(w_safety=0.0)),
    ("C1_force_ws0",    "prop", dict(w_safety=0.0, force_critical=True)),
]
# name: (reference, difference_required)
REFERENCE = {"C1_force": ("C0_weight", False),
             "C2_stopfb": ("C0_weight", True),
             "C3_force_stop": ("C0_weight", True),
             "C4_twotier": ("C3_force_stop", True),
             "C5_tighten_only": ("C0_weight", True),
             "B2_always_stop": ("B1_post", True),
             "C1_force_ws0": ("C0_ws0", True)}
if args.set == "onset" and not args.only:
    CONDITIONS = [c for c in CONDITIONS if c[0] in ("C0_weight", "C3_force_stop",
                                                    "B1_post")]
if args.only:
    keep = set(args.only.split(","))
    unknown = keep - {c[0] for c in CONDITIONS}
    assert not unknown, "unknown condition(s): %s" % sorted(unknown)
    CONDITIONS = [c for c in CONDITIONS if c[0] in keep]
CNAMES = [c[0] for c in CONDITIONS]
DUMP = set(args.dump.split(",")) if args.dump else set()

FLAGS = ("force_critical", "fallback_tube", "always_stop", "hard_barrier",
         "freeze_structure")
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


def seg_stats(H, kd, kf):
    """ferr in the three windows, mean speed in the healthy window, first violation."""
    out = dict(ferr_h=NAN, ferr_w=NAN, ferr_p=NAN, v_h=NAN, k_viol=-1)
    if "h" not in H:
        return out
    h = np.asarray(H["h"], float)
    if h.ndim == 1:
        h = h[:, None]
    T = h.shape[0]
    kf_ = T if (kf is None or kf < 0) else int(min(kf, T))
    kd_ = kf_ if (kd is None or kd < 0) else int(min(kd, kf_))
    e = np.abs(h - gap0)
    out.update(ferr_h=segmean(e[:kd_]), ferr_w=segmean(e[kd_:kf_]),
               ferr_p=segmean(e[kf_:]))
    if "v" in H:
        out["v_h"] = segmean(np.asarray(H["v"], float)[:kd_])
    hmin = np.where(np.isfinite(h), h, np.inf).min(axis=1)
    bad = np.where(hmin < 0)[0]
    out["k_viol"] = int(bad[0]) if bad.size else -1
    return out


# ---------------------------------------------------------------- run sets
base = Cfg(); el = Elements(base.plant.N); rc = base.risk
rho = 1.0 - rc.alpha / len(el)
gap0 = base.plant.spacing - base.plant.d_min
ENAME = {v: k for k, v in el.index.items()}

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
print("exp9 [%s] : %d runs x %d conditions" % (args.set, len(runs), len(CONDITIONS)))
print("=" * 96)
cr = build_calibration(base, Elements, base.plant.N, runs=30)

res = {name: [] for name in CNAMES}
meta = []
keepH = {}
n_det_mismatch = 0
n_viol_mismatch = 0
crit_values = set()
t0 = time.time()
for i, (lab, e_f, tr, seed, sh, cc, si) in enumerate(runs):
    c_plant = make_cfg(sh)
    assert c_plant.deg.t_deg > 0, "onset shift %+.1f makes t_deg <= 0" % sh
    pl = Plant(c_plant, el, failing=[e_f], rng=np.random.default_rng(seed))
    ks, R = risk_trace(pl, cr, rc)
    x0 = pl.x0()
    kf = pl.k_fail
    idx_f = el.index[e_f]
    det = detection_stats(ks, R, idx_f, rho, rc.persistence, kf, pl.dt, len(el))

    kflag = [first_flag(ks, R, rho, rc.persistence, j) for j in range(len(el))]
    k_true = kflag[idx_f]
    det_mine = (kf is not None) and 0 <= k_true <= kf
    if det_mine != bool(det["detected"]):
        n_det_mismatch += 1
    kf_cut = kf if kf is not None else 10 ** 9
    fa = [ENAME[j] for j in range(len(el)) if j != idx_f and 0 <= kflag[j] < kf_cut]
    kd = k_true if det_mine else kf
    meta.append(dict(label=lab, scenario=str(e_f), trial=tr, shift=sh, cc=cc,
                     si=si, detected=bool(det["detected"]), typ=e_f[0],
                     k_fail=-1 if kf is None else int(kf),
                     k_det=k_true if det_mine else -1,
                     lead=(kf - k_true) * pl.dt if det_mine else NAN,
                     fa=len(fa) > 0, fa_names=";".join(map(str, fa)),
                     t_deg=c_plant.deg.t_deg,
                     t_fail=NAN if kf is None else kf * pl.dt))

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
            print("R shape:", np.asarray(R).shape, " len(ks):", len(ks),
                  " k_fail:", kf, " dt:", pl.dt)
            sys.exit(0)
        ss = safety_stats(H)
        r = recovery_time(H, kf, pl.dt)
        sg = seg_stats(H, kd, kf)
        if (sg["k_viol"] >= 0) != (ss["nviol"] > 0):
            n_viol_mismatch += 1
        crit = H.get("crit_kept", -1)
        crit_values.add(crit if np.isscalar(crit) else str(crit))
        row = dict(viol=ss["nviol"] > 0, area=ss["viol_area"],
                   min_h=ss["min_h"], nslack=ss["nslack"],
                   rec=r, nsw=len(H["switch"]),
                   qpfail=H.get("qpfail", 0), crit=crit,
                   hrelax=H.get("hard_relaxed", 0), **sg)
        res[name].append(row)
        if name in DUMP and row["viol"] and cc and det["detected"]:
            keepH[(name, i)] = H
print("  done in %.0f s" % (time.time() - t0))

# ---------------------------------------------- diagnostics
print("  diag: own first_flag vs detection_stats 'detected' differ in %d / %d runs"
      % (n_det_mismatch, len(runs)))
print("  diag: own first violation (h<0) vs safety_stats nviol>0 differ in %d / %d"
      " run-conditions" % (n_viol_mismatch, len(runs) * len(CONDITIONS)))
print("  diag: crit_kept values returned by runner: %s" % sorted(map(str, crit_values)))

# ---------------------------------------------- check 1: reproduce exp8
if args.set == "cc30":
    if not os.path.exists("results/exp8_wsafety.csv"):
        print("  !! WARNING: results/exp8_wsafety.csv not found -- the exp8 "
              "reproduction check was SKIPPED (copy results/ from the old folder)")
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

# ---------------------------------------------- check 2: flags are wired
def sig(x):
    return (x["viol"], x["nslack"], x["nsw"], round(x["min_h"], 4))
for name, (refname, required) in REFERENCE.items():
    if name in res and refname in res:
        nd = sum(sig(a) != sig(b) for a, b in zip(res[name], res[refname]))
        note = ""
        if nd == 0:
            note = ("   !! flag NOT wired (a difference is required)" if required
                    else "   (expected: the weight already retains the edge here)")
        print("  wiring: %-16s differs from %-14s in %3d runs%s"
              % (name, refname, nd, note))

# ---------------------------------------------------------------- report
def col(name, k):
    return np.array([x[k] for x in res[name]], dtype=float)
def mcol(k, dtype=float):
    return np.array([m[k] for m in meta], dtype=dtype)
ccm = mcol("cc", bool); detm = mcol("detected", bool); fam = mcol("fa", bool)
typ = np.array([m["typ"] for m in meta]); shm = mcol("shift"); sim = mcol("si", int)
leadm = mcol("lead"); tdeg = mcol("t_deg"); tfail = mcol("t_fail")

if args.set == "c1":
    GROUPS = [("c1 main : scenarios 0-13, onset +0.0 s  (cc runs = cc30)", sim < 14),
              ("c1 repeat : scenarios 14-19, other seeds, onset +0.5 s", sim >= 14),
              ("c1 pooled : all %d runs" % len(meta), np.ones(len(meta), bool))]
elif args.set == "onset":
    GROUPS = [("onset shift %+.1f s" % s, shm == s) for s in sorted(set(shm))]
else:
    GROUPS = [("cc30", np.ones(len(meta), bool))]

HDR = ("%-16s %9s %15s %4s %6s %6s %7s %4s %7s %7s %7s %6s %6s"
       % ("condition", "viol(cc)", "Wilson 95%", "miss", "struct", "crit",
          "rec(cc)", "nr", "ferr_h", "ferr_w", "ferr_p", "v_h", "qpfail"))

for title, sel in GROUPS:
    m_cc = ccm & sel
    print("\n" + "-" * 96)
    print(title)
    # detector summary (condition-independent)
    d_cc = m_cc & detm
    print("  detector: t_deg %s s, t_fail %s s | detected %d/%d cc runs | "
          "lead (detected cc) mean %s s, min %s s | false alarm in %d/%d runs"
          % (",".join("%.1f" % x for x in sorted(set(tdeg[sel]))),
             ",".join("%.1f" % x for x in sorted(set(tfail[sel][np.isfinite(tfail[sel])]))),
             int(d_cc.sum()), int(m_cc.sum()),
             fx(nm(leadm[d_cc]), 0, 2), fx(np.nanmin(leadm[d_cc]) if d_cc.any() else NAN, 0, 2),
             int((fam & sel).sum()), int(sel.sum())))
    print("  windows over cc runs: ferr_h [0,k_det)  ferr_w [k_det,k_fail)  "
          "ferr_p [k_fail,T)   nr = not recovered (rec = nan)")
    print(HDR)
    for name in CNAMES:
        V = col(name, "viol").astype(bool)
        n = int(m_cc.sum()); k = int((V & m_cc).sum())
        lo, hi = wilson(k, n)
        miss = int((V & m_cc & ~detm).sum()); struct = k - miss
        cv = col(name, "crit")[m_cc & detm]; cv = cv[cv >= 0]
        crit_s = "%d/%d" % (int((cv == 1).sum()), cv.size) if cv.size else "n/a"
        rec = col(name, "rec")[m_cc]
        print("%-16s %4d/%-4d [%5.1f,%5.1f] %4d %6d %6s %s %4d %s %s %s %s %6d"
              % (name, k, n, lo, hi, miss, struct, crit_s,
                 fx(nm(rec), 7, 2), int(np.sum(~np.isfinite(rec))),
                 fx(nm(col(name, "ferr_h")[m_cc])), fx(nm(col(name, "ferr_w")[m_cc])),
                 fx(nm(col(name, "ferr_p")[m_cc])), fx(nm(col(name, "v_h")[m_cc]), 6, 3),
                 int(col(name, "qpfail")[sel].sum())))
    for name, (refname, _) in REFERENCE.items():
        if name in res and refname in res:
            a = col(refname, "viol").astype(bool) & m_cc
            b = col(name, "viol").astype(bool) & m_cc
            b10, b01, p = mcnemar(a, b)
            print("  McNemar %-16s vs %-14s (cc): improved %d, degraded %d, p=%.2e"
                  % (name, refname, b10, b01, p))
    if "C4_twotier" in res:
        print("  C4 hard barrier relaxed to soft in %d QP calls (total)"
              % int(col("C4_twotier", "hrelax")[sel].sum()))

    # healthy-window cost, split by false alarm (all runs of the group)
    print("  healthy-window cost (all runs in group):  ferr_h / v_h")
    for name in CNAMES:
        out = []
        for lab_, msk in (("no FA", sel & ~fam), ("FA", sel & fam)):
            out.append("%s n=%3d ferr_h %s v_h %s" % (
                lab_, int(msk.sum()), fx(nm(col(name, "ferr_h")[msk])),
                fx(nm(col(name, "v_h")[msk]), 6, 3)))
        print("    %-16s %s" % (name, "  |  ".join(out)))

    # per fault type (c1)
    if args.set == "c1":
        print("  by fault type:  n | rec mean (finite) | nr | ferr_p | ferr_h")
        for t in ("u", "v", "e"):
            mt = (typ == t) & sel
            if not mt.any():
                continue
            for name in CNAMES:
                rec = col(name, "rec")[mt]
                print("    %s  %-16s n=%3d  rec %s  nr %3d  ferr_p %s  ferr_h %s"
                      % (t, name, int(mt.sum()), fx(nm(rec), 6, 2),
                         int(np.sum(~np.isfinite(rec))),
                         fx(nm(col(name, "ferr_p")[mt])), fx(nm(col(name, "ferr_h")[mt]))))

# ---------------------------------------------- structural-route dump
dump_names = [n for n in CNAMES if n in DUMP]
if dump_names:
    print("\n" + "-" * 96)
    print("structural-route violations (cc, detected, violated)")
    print("%-16s %-14s %6s %6s %6s %7s %6s %7s %5s  %s"
          % ("condition", "run", "shift", "k_det", "k_fail", "lead s", "crit",
             "viol-f s", "nsw", "false alarms / first switches"))
    for name in dump_names:
        for i, (m, x) in enumerate(zip(meta, res[name])):
            if not (x["viol"] and m["cc"] and m["detected"]):
                continue
            dtv = ((x["k_viol"] - m["k_fail"]) * (m["t_fail"] / m["k_fail"])
                   if x["k_viol"] >= 0 and m["k_fail"] > 0 else NAN)
            sw = keepH.get((name, i), {}).get("switch", [])
            print("%-16s %-14s %+6.1f %6d %6d %s %6s %s %5d  FA[%s] sw%s"
                  % (name, m["label"], m["shift"], m["k_det"], m["k_fail"],
                     fx(m["lead"], 7, 2), str(int(x["crit"])), fx(dtv, 7, 2),
                     int(x["nsw"]), m["fa_names"], list(sw)[:4]))
    pk = "results/exp9_%s_struct_runs.pkl" % args.set
    try:
        with open(pk, "wb") as f:
            pickle.dump({"%s|%s|%+.1f" % (n, meta[i]["label"], meta[i]["shift"]): H
                         for (n, i), H in keepH.items()}, f)
        print("  saved %d histories to %s" % (len(keepH), pk))
    except Exception as ex:
        print("  !! could not pickle histories: %s" % ex)

# ---------------------------------------------------------------- save
out = {"cc": ccm, "detected": detm, "shift": shm, "si": sim, "typ": typ,
       "fa": fam, "lead": leadm, "t_deg": tdeg, "t_fail": tfail,
       "k_fail": mcol("k_fail", int), "k_det": mcol("k_det", int),
       "labels": np.array([m["label"] for m in meta])}
for name in CNAMES:
    for k in ("viol", "area", "min_h", "nslack", "rec", "nsw", "qpfail", "crit",
              "hrelax", "ferr_h", "ferr_w", "ferr_p", "v_h", "k_viol"):
        out["%s__%s" % (name, k)] = col(name, k)
fn = "results/exp9_%s.npz" % args.set
np.savez(fn, **out)
print("\nsaved %s  sha256 %s" % (fn, hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]))

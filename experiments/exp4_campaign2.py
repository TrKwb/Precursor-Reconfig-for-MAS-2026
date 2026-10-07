## -*- coding: utf-8 -*-
"""Campaign II: the scaling study, re-run on a scenario set that is shared
with an N = 4 result already in the paper, and checked against it.

Two reference sets are supported (choose with --ref):

  --ref=exp8  (default)  3 actuator-fault scenarios (u,1,1) (u,2,2) (u,3,3)
                         x 10 seeds, seed rule of exp8_wsafety.py.  At N = 4
                         the proposed scheme must reproduce the w_s = 2.0
                         column of Figure 4 RUN FOR RUN (violation, area,
                         recovery), asserted against results/exp8_wsafety.csv.
  --ref=exp1             the 50 collision-capable runs of Campaign I
                         (scenario indices 1, 2, 3 and the shifted-onset
                         repeats 14, 15 of exp1_main.py, seed rule
                         10000 + 131*si + tr).  At N = 4 all three methods must
                         reproduce results/exp1.npz run for run.

The scenario set is held FIXED across N = 4, 8, 16 (it does not grow with N),
so the study measures the effect of enlarging the system around a fixed set of
faults.  Nothing is saved if the assertion fails.

Recorded per run, with explicit denominators:
  * violation / area / min h / slack count / switches / recovery, per method
  * per-RUN detection of the failing element before failure, and lead time
  * per-EVALUATION false alarms on the healthy elements, raw (r_e > rho, the
    event Proposition 1 bounds) and after the n_p persistence rule
  * selection time: 400 random risk profiles, 5 warm-up calls discarded

The three methods run on deep copies of ONE plant object, so they share x(0),
disturbance and diagnostic realisation exactly.

Usage : python experiments/exp4_campaign2.py [--ref=exp8|exp1] [--dry]
        --dry : skip the assertion; output is marked UNVERIFIED
Output: results/exp4_campaign2.npz (+ .sha256)
"""
import sys
import os
import csv
import copy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
from masrecon.config import Cfg
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
from masrecon.metrics import safety_stats, recovery_time
from _util import (wilson, clopper_pearson, mcnemar_exact, fisher_exact,
                   cochran_armitage, matthews, persistence_flags,
                   time_selection, write_hash, env_info)

DRY = "--dry" in sys.argv
REF = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--ref=")), "exp8")
assert REF in ("exp8", "exp1"), "--ref must be exp8 or exp1"
NS = (4, 8, 16)
NSEED = 10
WS_REF = 2.0                         # w_safety of the paper's operating point
METHODS = ("fixed", "post", "prop")
OUT = "results/exp4_campaign2.npz"
os.makedirs("results", exist_ok=True)


# ======================================================================
#  Scenario sets.  Each entry: (index used by the seed rule, failing list,
#  onset shift [s] added to cfg.deg.t_deg).
# ======================================================================
# --- exp8 set: identical to exp8_wsafety.py -------------------------------
SCENARIOS_EXP8 = [(0, [("u", 1, 1)], 0.0),
                  (1, [("u", 2, 2)], 0.0),
                  (2, [("u", 3, 3)], 0.0)]


def seed_exp8(si, tr):
    return 10000 + 131 * (si + 1) + tr


# --- exp1 set: the collision-capable scenarios of exp1_main.py ------------
#     SC[1..3] = (u,1,1) (u,2,2) (u,3,3);  SC[14] = (u,1,1), SC[15] = (u,3,3)
#     with onset shifted by +0.5 s (exp1_main.py: "if si >= 14").
SCENARIOS_EXP1 = [(1, [("u", 1, 1)], 0.0),
                  (2, [("u", 2, 2)], 0.0),
                  (3, [("u", 3, 3)], 0.0),
                  (14, [("u", 1, 1)], 0.5),
                  (15, [("u", 3, 3)], 0.5)]


def seed_exp1(si, tr):
    return 10000 + 131 * si + tr


SCENARIOS, SEED = ((SCENARIOS_EXP8, seed_exp8) if REF == "exp8"
                   else (SCENARIOS_EXP1, seed_exp1))
SHARE_X0 = REF == "exp1"             # replicate exp1_main.py exactly


def make_cfg(N, shift=0.0):
    cfg = Cfg()
    cfg.plant.N = N
    cfg.w_safety = WS_REF
    if shift:
        cfg.deg.t_deg = cfg.deg.t_deg + shift
    return cfg
# ======================================================================


def check_candidate_graph(el, N):
    links = [e for e in el.all if e[0] == "e"]
    miss = [("e", i, i + 1) for i in range(N - 1) if ("e", i, i + 1) not in el.index]
    hop = max(abs(e[2] - e[1]) for e in links)
    print("  |C| = %d : %d links (complete graph %d), max hop %d"
          % (len(el), len(links), N * (N - 1) // 2, hop))
    assert not miss, ("safety-critical edges missing from the candidate set at "
                      "N=%d: %s" % (N, miss))
    return len(links), hop


def one_size(N):
    base = make_cfg(N)
    el = Elements(N)
    nlink, hop = check_candidate_graph(el, N)
    rho = 1.0 - base.risk.alpha / len(el)
    npers = base.risk.persistence
    print("  calibrating ...", flush=True)
    cr = build_calibration(base, Elements, N, runs=30)

    R_ = {m: dict(viol=[], area=[], minh=[], nslack=[], nsw=[], rec=[])
          for m in METHODS}
    det, lead, lab = [], [], []
    raw_k = raw_n = flg_k = flg_n = 0
    for si, fail, shift in SCENARIOS:
        c = make_cfg(N, shift)
        for tr in range(NSEED):
            pl = Plant(c, el, failing=fail,
                       rng=np.random.default_rng(SEED(si, tr)))
            ks, R = risk_trace(pl, cr, c.risk)
            ks = np.asarray(ks); R = np.asarray(R)
            lab.append("%s%s/%d" % ("+".join("%s%d" % (e[0], e[1]) for e in fail),
                                    "s" if shift else "", tr))
            # --- detection (per run) and false alarms (per evaluation)
            jf = [el.index[e] for e in fail]
            hl = [j for j in range(len(el)) if j not in jf]
            over = R > rho
            fl = persistence_flags(over, npers)
            hit = np.flatnonzero(fl[:, jf].any(axis=1) & (ks < pl.k_fail))
            det.append(hit.size > 0)
            lead.append((pl.k_fail - ks[hit[0]]) * pl.dt if hit.size else np.nan)
            raw_k += int(over[:, hl].sum()); raw_n += over[:, hl].size
            flg_k += int(fl[npers - 1:, hl].sum()); flg_n += fl[npers - 1:, hl].size
            # --- closed loop: identical copies => identical realisation
            kw = {"x0": pl.x0()} if SHARE_X0 else {}
            for m in METHODS:
                H = run(copy.deepcopy(pl), el, c, m, ks, R, rho, **kw)
                ss = safety_stats(H)
                r = recovery_time(H, pl.k_fail, pl.dt)
                d = R_[m]
                d["viol"].append(ss["nviol"] > 0); d["area"].append(ss["viol_area"])
                d["minh"].append(ss["min_h"]); d["nslack"].append(ss["nslack"])
                d["nsw"].append(len(H["switch"]))
                d["rec"].append(np.nan if r != r else r)
    sel = time_selection(el, base, N, rho, np.random.default_rng(777 + N))
    out = {"nC": len(el), "nlink": nlink, "hop": hop, "alpha": base.risk.alpha,
           "npers": npers, "label": np.array(lab),
           "det": np.array(det), "lead": np.array(lead),
           "far_raw_k": raw_k, "far_raw_n": raw_n,
           "far_flag_k": flg_k, "far_flag_n": flg_n, "sel_ms": sel}
    for m in METHODS:
        for k, v in R_[m].items():
            out["%s_%s" % (m, k)] = np.array(v)
    return out


# ---------------------------------------------------------------- assertions
def assert_reproduces_exp8(r, ws=WS_REF):
    """N = 4 must reproduce the w_s column of Figure 4 run for run."""
    src = "results/exp8_wsafety.csv"
    with open(src, encoding="utf-8") as f:
        rows = [x for x in csv.DictReader(f) if abs(float(x["w_safety"]) - ws) < 1e-9]
    v = np.array([int(x["violation"]) for x in rows], bool)
    a = np.array([float(x["area"]) for x in rows])
    rc = np.array([float(x["rec"]) if x["rec"] else np.nan for x in rows])
    ns = np.array([int(x["nslack"]) for x in rows])
    assert v.size == r["prop_viol"].size, (
        "run count %d vs %d rows of %s at w_s=%g" % (r["prop_viol"].size, v.size, src, ws))
    bad = np.flatnonzero(v != r["prop_viol"])
    assert bad.size == 0, "violations differ from exp8 at runs %s" % r["label"][bad].tolist()
    bad = np.flatnonzero(~np.isclose(a, r["prop_area"], atol=1e-4))
    assert bad.size == 0, "areas differ from exp8 at runs %s" % r["label"][bad].tolist()
    bad = np.flatnonzero(~np.isclose(rc, r["prop_rec"], atol=1e-3, equal_nan=True))
    assert bad.size == 0, "recovery differs from exp8 at runs %s" % r["label"][bad].tolist()
    nd = np.flatnonzero(ns != r["prop_nslack"])
    if nd.size:
        print("  note: slack count differs from exp8 at %s (not asserted)"
              % r["label"][nd].tolist())
    print("  ASSERTION PASSED: N = 4 reproduces Figure 4 (w_s = %g) run for run" % ws)


def assert_reproduces_table1(r):
    """N = 4 must reproduce the collision-capable subset of Table 1 run for
    run, for all three methods.  exp1.npz stores runs in scenario order, so the
    collision-capable mask selects them in the same order as SCENARIOS_EXP1."""
    ref = np.load("results/exp1.npz")
    cap = ref["cc"].astype(bool)
    hint = (" -- SCENARIOS_EXP1/seed_exp1 do not match exp1_main.py, or "
            "exp1.npz predates the deep-copy fix (re-run exp1_main.py)")
    assert cap.sum() == r["post_viol"].size, (
        "run count %d vs %d collision-capable runs in exp1.npz"
        % (r["post_viol"].size, cap.sum()) + hint)
    for m in METHODS:
        v = ref["%s_anyviol" % m][cap].astype(bool)
        bad = np.flatnonzero(v != r["%s_viol" % m])
        assert bad.size == 0, ("%s: violations differ at runs %s"
                               % (m, r["label"][bad].tolist()) + hint)
        bad = np.flatnonzero(~np.isclose(ref["%s_viol_area" % m][cap],
                                         r["%s_area" % m], atol=1e-4))
        assert bad.size == 0, ("%s: violation areas differ at runs %s"
                               % (m, r["label"][bad].tolist()) + hint)
        if m != "fixed":
            bad = np.flatnonzero(~np.isclose(ref["%s_rec" % m][cap], r["%s_rec" % m],
                                             atol=1e-3, equal_nan=True))
            assert bad.size == 0, ("%s: recovery differs at runs %s"
                                   % (m, r["label"][bad].tolist()) + hint)
    nd = np.flatnonzero(ref["prop_nslack"][cap] != r["prop_nslack"])
    if nd.size:
        print("  note: slack count differs from exp1 at %s (not asserted)"
              % r["label"][nd].tolist())
    print("  ASSERTION PASSED: N = 4 reproduces Table 1 (collision-capable subset) "
          "run for run")


# ======================================================================
print("=" * 78)
print("exp4_campaign2 : reference set = %s%s" % (REF, "  [--dry]" if DRY else ""))
print(env_info())
print("=" * 78)
res = {}
for N in NS:
    print("\nN = %d" % N)
    res[N] = one_size(N)
    if N == 4 and not DRY:
        (assert_reproduces_exp8 if REF == "exp8" else assert_reproduces_table1)(res[N])

# ------------------------------------------------------------- summary table
W0 = 34
def row(name, f):
    print("%-*s" % (W0, name) + "".join("%22s" % f(res[N], N) for N in NS))

aE4 = res[NS[0]]["alpha"] / res[NS[0]]["nC"]
print("\n" + "-" * (W0 + 22 * len(NS)))
row("", lambda r, N: "N=%d" % N)
row("candidate elements |C| (links)", lambda r, N: "%d (%d)" % (r["nC"], r["nlink"]))
row("runs", lambda r, N: "%d" % r["det"].size)
row("nominal alpha_e", lambda r, N: "%.5f" % (r["alpha"] / r["nC"]))
row("detection per run", lambda r, N: "%d/%d = %.1f%%" % (r["det"].sum(), r["det"].size, 100 * r["det"].mean()))
row("  95% CI (Clopper-Pearson)", lambda r, N: "[%.0f, %.0f]" % tuple(100 * np.array(clopper_pearson(r["det"].sum(), r["det"].size))))
row("raw FAR per evaluation", lambda r, N: "%.5f" % (r["far_raw_k"] / r["far_raw_n"]))
row("  n (evaluations)", lambda r, N: "%d" % r["far_raw_n"])
row("  raw FAR / alpha_e", lambda r, N: "%.2f" % (r["far_raw_k"] / r["far_raw_n"] / (r["alpha"] / r["nC"])))
row("flag FAR after n_p rule", lambda r, N: "%.5f" % (r["far_flag_k"] / r["far_flag_n"]))
row("iso-alpha_e family-wise level", lambda r, N: "%.2f" % (aE4 * r["nC"]))
row("mean lead time [s]", lambda r, N: "%.2f" % np.nanmean(r["lead"]))
for m in METHODS:
    row("violation, %s" % m, lambda r, N, m=m: "%d/%d = %.0f%%" % (r[m + "_viol"].sum(), r[m + "_viol"].size, 100 * r[m + "_viol"].mean()))
row("  Wilson 95% CI, proposed", lambda r, N: "[%.1f, %.1f]" % tuple(100 * np.array(wilson(r["prop_viol"].sum(), r["prop_viol"].size))))

def _bc(r):
    return (int(np.sum(r["post_viol"] & ~r["prop_viol"])),
            int(np.sum(~r["post_viol"] & r["prop_viol"])))
row("McNemar b, c", lambda r, N: "b=%d, c=%d" % _bc(r))
row("  exact p", lambda r, N: "%.1e" % mcnemar_exact(*_bc(r)))

def _carea(r, m):
    v = r[m + "_viol"]
    return r[m + "_area"][v].mean() if v.any() else 0.0
row("mean area, post / prop", lambda r, N: "%.3f / %.3f" % (r["post_area"].mean(), r["prop_area"].mean()))
row("  given a violation", lambda r, N: "%.3f / %.3f" % (_carea(r, "post"), _carea(r, "prop")))
row("violations that are misses", lambda r, N: "%d/%d" % (np.sum(r["prop_viol"] & ~r["det"]), r["prop_viol"].sum()))

def _rec(r, m):
    x = r[m + "_rec"]; x = x[~np.isnan(x)]
    return "%.2f+-%.2f (%d val.)" % (x.mean(), x.std(), np.unique(np.round(x, 3)).size) if x.size else "n/a"
row("recovery, post [s]", lambda r, N: _rec(r, "post"))
row("recovery, prop [s]", lambda r, N: _rec(r, "prop"))
row("selection/elem median [us]", lambda r, N: "%.2f" % (1e3 * np.median(r["sel_ms"]) / r["nC"]))
row("  [p5, p95] [us]", lambda r, N: "[%.2f, %.2f]" % tuple(1e3 * np.percentile(r["sel_ms"], [5, 95]) / r["nC"]))

V = np.concatenate([res[N]["prop_viol"] for N in NS])
M = ~np.concatenate([res[N]["det"] for N in NS])
tab = (int(np.sum(V & M)), int(np.sum(V & ~M)), int(np.sum(~V & M)), int(np.sum(~V & ~M)))
print("\npooled violation vs miss: [[viol&miss %d, viol&det %d], [ok&miss %d, ok&det %d]]" % tab)
print("  MCC = %.2f, Fisher exact p = %.1e  (%d runs)" % (matthews(V, M), fisher_exact(*tab), V.size))
z, p = cochran_armitage([res[N]["prop_viol"].sum() for N in NS],
                        [res[N]["prop_viol"].size for N in NS])
print("trend of proposed violations with N: Cochran-Armitage z = %.2f, p = %.1e" % (z, p))
z, p = cochran_armitage([(~res[N]["det"]).sum() for N in NS],
                        [res[N]["det"].size for N in NS])
print("trend of detection misses with N  : Cochran-Armitage z = %.2f, p = %.1e" % (z, p))

# --------------------------------------------------------- per-run diagnostics
print("\nviolating runs of the proposed scheme:")
print("  %3s %-11s %9s %8s %5s %7s %4s %7s" % ("N", "run", "area", "min h", "det", "nslack", "nsw", "rec"))
for N in NS:
    r = res[N]
    for i in np.flatnonzero(r["prop_viol"]):
        print("  %3d %-11s %9.3f %8.4f %5s %7d %4d %7.2f"
              % (N, r["label"][i], r["prop_area"][i], r["prop_minh"][i], bool(r["det"][i]),
                 r["prop_nslack"][i], r["prop_nsw"][i], r["prop_rec"][i]))
    big = np.flatnonzero(r["prop_viol"] & (r["prop_area"] > 2 * np.nanmax(r["post_area"])))
    if big.size:
        print("  !! N=%d: %d proposed run(s) with area > 2 x the worst post-failure area" % (N, big.size))
    for m in ("post", "prop"):
        x = r[m + "_rec"]; x = x[~np.isnan(x)]
        if x.size > 1 and np.unique(np.round(x, 3)).size == 1:
            print("  !! N=%d: recovery of '%s' takes the single value %.3f s in all %d runs "
                  "(metric saturated)" % (N, m, x[0], x.size))

# -------------------------------------------------------------------- save
flat = {"NS": np.array(NS), "verified": np.array(not DRY), "ref": np.array(REF),
        "w_safety": np.array(WS_REF), "env": np.array(env_info())}
for N in NS:
    for k, v in res[N].items():
        flat["N%d_%s" % (N, k)] = np.asarray(v)
np.savez(OUT, **flat)
print("\nsaved %s  sha256 %s%s" % (OUT, write_hash(OUT)[:16],
                                   "  [UNVERIFIED: --dry]" if DRY else "  [verified against %s]" % REF))
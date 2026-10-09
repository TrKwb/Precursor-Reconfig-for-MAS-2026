# -*- coding: utf-8 -*-
"""CHECK items of main.tex v12.  Usage: python -X utf8 tools/check_impl.py"""
import os, re, sys, glob, inspect, copy, importlib
sys.path.insert(0, os.getcwd())
import numpy as np
SRC = sorted(glob.glob("masrecon/*.py"))

def head(t): print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)
def grep(pat, files=SRC, ctx=2, maxhits=12):
    n = 0
    for f in files:
        if not os.path.exists(f): continue
        L = open(f, encoding="utf-8").read().splitlines()
        for i, l in enumerate(L):
            if re.search(pat, l):
                print("--- %s:%d" % (f, i + 1))
                for j in range(max(0, i - ctx), min(len(L), i + ctx + 1)):
                    print("%5d%s %s" % (j + 1, ">" if j == i else " ", L[j]))
                n += 1
                if n >= maxhits: print("  ... (truncated)"); return n
    if n == 0: print("  (no match for %r)" % pat)
    return n

from masrecon.config import Cfg
head("[A] condition settings: defined in Cfg? read anywhere in masrecon/?")
KEYS = ["w_safety", "policy_T", "nonretreat", "hard_barrier", "freeze_structure",
        "latch_release", "force_critical", "fallback_tube", "nonretreat_scope",
        "latch_soft", "soft_barrier"]
c0 = Cfg()
for k in KEYS:
    reads = [os.path.basename(f) for f in SRC if not f.endswith("config.py")
             and re.search(r"[\"'.]%s\b" % k, open(f, encoding="utf-8").read())]
    print("%-18s in Cfg: %-5s default=%-12r read in: %s" % (
        k, hasattr(c0, k), getattr(c0, k, None),
        ", ".join(reads) or "NOWHERE  <-- silently ignored"))
grep(r"T_latch_soft|T_latch_rel10|T_latch_stop\b", ["experiments/exp9_remedies.py"], ctx=1)

head("[B] shield: functions mentioning shield / F_stop")
for f in SRC:
    mod = importlib.import_module("masrecon." + os.path.basename(f)[:-3])
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ != mod.__name__ or name == "run": continue
        s = inspect.getsource(fn)
        if re.search(r"shield|F_stop|stop_safe|in_F|brake", s):
            print("### %s.%s  (%d lines)" % (mod.__name__, name, s.count("\n")))
            print(s[:3500])
grep(r"shield", ["masrecon/runner.py"], ctx=3, maxhits=15)

head("[D] back-off sigma vs r_bar")
grep(r"sigma|back.?off|backoff", ctx=2, maxhits=15)

head("[E] process disturbance in plant / runner")
grep(r"normal\(|disturb|noise|w_proc|process", ["masrecon/plant.py", "masrecon/runner.py"], ctx=2)

head("[F] nonretreat_scope / prefail")
grep(r"nonretreat_scope|prefail")

head("[C,E] runtime: one cc run of the proposed scheme")
from masrecon.structure import Elements
from masrecon.plant import Plant
from masrecon.risk import build_calibration, risk_trace
from masrecon.runner import run
base = Cfg(); N = base.plant.N; el = Elements(N); rc = base.risk
rho = 1.0 - rc.alpha / len(el); dt = base.plant.dt
cr = build_calibration(base, Elements, N, runs=30)
e_f = ("u", 2, 2); jf = 2
pl = Plant(Cfg(), el, failing=[e_f], rng=np.random.default_rng(10000 + 131 * 2 + 0))
ks, Rk = risk_trace(pl, cr, rc); x0 = pl.x0(); kf = pl.k_fail
c2 = Cfg()
for k, v in dict(w_safety=2.0, policy_T=True, nonretreat=True, hard_barrier=True,
                 freeze_structure=True, latch_release=10).items():
    setattr(c2, k, v)
H = run(copy.deepcopy(pl), el, c2, "prop", np.asarray(ks), np.asarray(Rk, float), rho, x0=x0)
print("keys of H:")
for k in sorted(H):
    try:
        a = np.asarray(H[k]); d = "shape %s dtype %s" % (a.shape, a.dtype)
    except Exception:
        d = type(H[k]).__name__
    print("  %-16s %s" % (k, d))
sh = np.asarray(H["shield"])
print("shield: sum(>0) = %d ; ndim %d" % (int((sh > 0).sum()), sh.ndim))
if sh.ndim == 2: print("  per agent:", (sh > 0).sum(axis=0))
print("hard_relaxed:", repr(H.get("hard_relaxed"))[:200])
print("crit_kept   :", repr(H.get("crit_kept"))[:200])
cp = base.plant
cd = next((getattr(cp, n) for n in ("c", "drag", "c_drag", "cdrag") if hasattr(cp, n)), None)
if all(k in H for k in ("p", "v", "u")) and cd is not None:
    p, v, u = (np.asarray(H[k], float) for k in ("p", "v", "u"))
    T = min(len(p) - 1, len(u))
    rp = p[1:T + 1] - (p[:T] + dt * v[:T] + 0.5 * dt * dt * u[:T])
    rv = v[1:T + 1] - ((1 - cd * dt) * v[:T] + dt * u[:T])
    ok = np.ones_like(rp, bool); ok[kf:, jf] = False          # dead actuator: commanded != applied
    print("dynamics residual (healthy agent-steps): max|dp| %.2e  max|dv| %.2e  (0 => no disturbance)"
          % (np.abs(rp[ok]).max(), np.abs(rv[ok]).max()))
else:
    print("cannot check residual: need H['p'], H['v'], H['u'] and drag in cfg.plant (c=%r)" % cd)

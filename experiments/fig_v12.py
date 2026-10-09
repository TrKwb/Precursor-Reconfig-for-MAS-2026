# -*- coding: utf-8 -*-
"""Regenerate figures for main.tex v12:  python -X utf8 experiments/fig_v12.py [all|fig1|fig2|fig5|fig8]
Originals are copied once to figs/old_v11/."""
import sys, os, copy, math, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ap = argparse.ArgumentParser(); ap.add_argument("which", nargs="?", default="all")
ap.add_argument("--fig2-run", type=int, default=10, help="index into cc30 runs")
args = ap.parse_args()
os.makedirs("figs/old_v11", exist_ok=True); os.makedirs("results", exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})

def save(fig, name):
    fn = "figs/%s.pdf" % name
    if os.path.exists(fn) and not os.path.exists("figs/old_v11/%s.pdf" % name):
        shutil.copy(fn, "figs/old_v11/%s.pdf" % name)
    fig.savefig(fn, bbox_inches="tight"); fig.savefig(fn[:-4] + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", fn)

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

_S = {}
def setup():
    if _S: return _S
    from masrecon.config import Cfg
    from masrecon.structure import Elements
    from masrecon.plant import Plant
    from masrecon.risk import build_calibration, risk_trace
    from masrecon.runner import run
    from masrecon.metrics import safety_stats, recovery_time
    base = Cfg(); N = base.plant.N; el = Elements(N); rc = base.risk
    _S.update(Cfg=Cfg, Plant=Plant, run=run, risk_trace=risk_trace, safety_stats=safety_stats,
              recovery_time=recovery_time, el=el, rc=rc, N=N, dt=base.plant.dt,
              rho=1.0 - rc.alpha / len(el), cr=build_calibration(base, Elements, N, runs=30))
    return _S
def mk(S, **over):
    c = S["Cfg"]()
    for k, v in over.items(): setattr(c, k, v)
    return c
def cc_runs(N):
    return [("%s/%d" % (e, tr), e, 10000 + 131 * (si + 1) + tr)
            for si, e in enumerate([("u", i, i) for i in range(1, min(N, 4))]) for tr in range(10)]
def first_flag(ks, R, rho, P, j):
    cnt = 0
    for i, a in enumerate(np.asarray(R)[:, j] > rho):
        cnt = cnt + 1 if a else 0
        if cnt >= P: return int(ks[i])
    return -1
def prep(S, e_f, seed):
    pl = S["Plant"](mk(S), S["el"], failing=[e_f], rng=np.random.default_rng(seed))
    ks, R = S["risk_trace"](pl, S["cr"], S["rc"])
    return pl, np.asarray(ks), np.asarray(R, float), pl.x0()

TP = dict(w_safety=2.0, policy_T=True, nonretreat=True, hard_barrier=True)
PROP = dict(TP, freeze_structure=True, latch_release=10)

# ------------------------------------------------------------------ fig 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    def box(x, y, w, h, txt, fc="white", ls="-", fs=6.8, color="k"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=fc, ec="k", lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=color, linespacing=1.3)
    box(2, 41.5, 96, 3.3, r"environment: link loss $\cdot$ sensor fault $\cdot$ actuator degradation and failure",
        fc="#eeeeee", fs=7.2)
    X, W = [2, 35.5, 69], 29
    heads = ["A  precursor risk", "B  structure selection", "C  contract and control"]
    bodies = [
        "diagnostic signals $y_e(k)$\nPE / RQA window features\n$\\downarrow$\n"
        "split-conformal limit $\\bar r_e(k)$\nflag if $\\bar r_e>1-\\alpha_e$\n($n_p$ consecutive times)",
        "admissible set $A(\\alpha)$\n(downward closed)\n$\\downarrow$\ngraphic matroid,\n"
        "critical edges forced\nby contraction (Kruskal)\n$\\downarrow$\nstructure $s^\\star=(E,V,U)$",
        "contract per pair:\nenvelope or stop\nflag $\\Rightarrow$ latch to stop,\n"
        "release after $n_r$ unflagged\n$\\downarrow$\nDMPC + CBF: performance\nshield on $F_\\beta$: safety"]
    guar = ["healthy element excluded\nw.p. $\\leq\\alpha$ per evaluation\n(Prop. 2)",
            "exact, polynomial time (Prop. 3)\n(T4): structure must not\nvoid the contract",
            "safe if follower reaches\n$F_{\\mathrm{stop}}$ before failure\n(Thm. 1)"]
    for x, hd, bd, g in zip(X, heads, bodies, guar):
        box(x, 34.5, W, 4.2, hd, fc="#333333", fs=7.8, color="white")
        box(x, 13.0, W, 19.5, bd)
        box(x, 1.5, W, 9.5, g, ls="--", fs=6.5)
    for x0 in (X[0] + W, X[1] + W):
        ax.annotate("", xy=(x0 + 4.2, 22.7), xytext=(x0 + 0.4, 22.7),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color="k"))
    save(fig, "fig1_concept")

# ------------------------------------------------------------------ fig 2
def fig2():
    S = setup(); el, dt, rho = S["el"], S["dt"], S["rho"]
    lab, e_f, seed = cc_runs(S["N"])[args.fig2_run]
    pl, ks, R, x0 = prep(S, e_f, seed); kf = pl.k_fail; jf = e_f[1]; i_f = jf - 1; idx = el.index[e_f]
    kd = first_flag(ks, R, rho, S["rc"].persistence, idx)
    if not (0 <= kd <= kf): print("WARNING: run %s not detected; choose another --fig2-run" % lab)
    Hs = {nm: S["run"](copy.deepcopy(pl), el, mk(S, **ov), mt, ks, R, rho, x0=x0)
          for nm, mt, ov in (("fixed", "fixed", dict(w_safety=2.0)),
                             ("post-oracle", "post", dict(w_safety=2.0)),
                             ("proposed", "prop", PROP))}
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 5.0), sharex=True)
    for j in range(R.shape[1]):
        if j != idx: axs[0].plot(ks * dt, R[:, j], color="0.82", lw=0.5)
    axs[0].plot(ks * dt, R[:, idx], "k", lw=1.1, label="failing element")
    axs[0].axhline(rho, ls="--", color="k", lw=0.7, label=r"$1-\alpha_e$")
    axs[0].set_ylabel(r"$\bar r_e$"); axs[0].legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9)
    H = Hs["proposed"]; lat = np.asarray(H["latched"]).astype(bool)
    axs[1].step(np.arange(lat.shape[0]) * dt, lat[:, jf].astype(int), where="post", color="k", lw=1)
    sh = np.asarray(H["shield"]); shv = sh[:, i_f] if sh.ndim == 2 else sh
    k_sh = np.flatnonzero(np.asarray(shv) > 0)
    axs[1].plot(k_sh * dt, np.full(len(k_sh), 0.5), "|", color="0.4", ms=7, label="shield intervention")
    axs[1].set_yticks([0, 1]); axs[1].set_yticklabels(["env", "stop"]); axs[1].set_ylim(-0.3, 1.3)
    axs[1].set_ylabel("contract"); axs[1].legend(loc="center left", fontsize=6.5, frameon=True, framealpha=0.9)
    lo = 0
    for nm, st in (("fixed", ":"), ("post-oracle", "--"), ("proposed", "-")):
        hm = np.asarray(Hs[nm]["h"], float).min(axis=1); lo = min(lo, hm.min())
        axs[2].plot(np.arange(len(hm)) * dt, hm, st, color="k", lw=1, label=nm)
    axs[2].axhspan(lo - 0.2, 0, color="0.9", lw=0); axs[2].set_ylim(lo - 0.2, None)
    axs[2].set_ylabel(r"$\min_i h_i$ [m]"); axs[2].set_xlabel("time $t$ [s]")
    axs[2].legend(loc="lower left", fontsize=6.5, frameon=False)
    for a in axs:
        a.axvline(kf * dt, color="k", lw=0.6)
        if 0 <= kd <= kf: a.axvline(kd * dt, color="k", lw=0.6, ls="-.")
    for a, t in zip(axs, "abc"): a.text(0.01, 1.02, "(%s)" % t, transform=a.transAxes, weight="bold")
    print("fig2 run %s  lead %.2f s  min h: %s" % (lab, (kf - kd) * dt if kd >= 0 else float("nan"),
          {nm: round(float(np.asarray(Hs[nm]["h"]).min()), 3) for nm in Hs}))
    save(fig, "fig2_timeseries")

# ------------------------------------------------------------------ fig 5
def fig5():
    files = {4: "results/exp11_lead.npz", 8: "results/exp11_lead_N8.npz", 16: "results/exp11_lead_N16.npz"}
    Ns = [4, 8, 16]; prop, abl, det = [], [], []
    for N in Ns:
        d = np.load(files[N]); c = d["cond"].astype(str); l = d["lead"].astype(str)
        v = d["viol"].astype(bool); la = d["lead_act"].astype(float); m = l == "real"
        for name, out in (("T_latch_rel10", prop), ("C0_weight", abl)):
            s = m & (c == name); out.append((int(v[s].sum()), int(s.sum())))
        s = m & (c == "T_latch_rel10"); det.append((int((la[s] > 0).sum()), int(s.sum())))
    print("fig5 proposed", prop, " ablation", abl, " detection", det)
    FARr = [0.0147 / 0.0143, 0.0031 / 0.0045, 0.00013 / 0.0013]          # Table 5
    ISO = [0.20, 0.63, 2.17]
    SEL = [(1.22, 1.16, 1.29), (1.07, 1.05, 1.13), (0.99, 0.98, 1.08)]    # Table 5 / exp4
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9)); x = np.arange(3); w = 0.27
    for k, (lab_, data, hatch, fc) in enumerate((("post-oracle", [(30, 30)] * 3, "////", "white"),
                                                 ("ablation", abl, "", "0.7"),
                                                 ("proposed", prop, "", "0.15"))):
        W_ = [wilson(a, b) for a, b in data]
        axs[0].bar(x + (k - 1) * w, [q[0] for q in W_], w, color=fc, hatch=hatch, ec="k", lw=0.6, label=lab_,
                   yerr=[[q[0] - q[1] for q in W_], [q[2] - q[0] for q in W_]], capsize=2,
                   error_kw=dict(lw=0.6))
    axs[0].set_ylabel("runs with violation [%]"); h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, fontsize=6.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
    Wd = [wilson(a, b) for a, b in det]
    axs[1].errorbar(x, [q[0] for q in Wd], yerr=[[q[0] - q[1] for q in Wd], [q[2] - q[0] for q in Wd]],
                    fmt="o-", color="k", ms=3, capsize=2, lw=0.8)
    axs[1].set_ylabel("detection per run [%]"); axs[1].set_ylim(0, 105)
    a2 = axs[1].twinx(); a2.plot(x, FARr, "s--", color="0.5", ms=3, lw=0.8); a2.set_ylim(0, 1.5)
    a2.set_ylabel(r"FAR$/\alpha_e$", color="0.4")
    axs[2].bar(x, ISO, 0.5, color="0.6", ec="k", lw=0.6); axs[2].axhline(1, ls="--", color="k", lw=0.7)
    axs[2].set_ylabel(r"iso-$\alpha_e$ level")
    axs[3].errorbar(x, [s[0] for s in SEL], yerr=[[s[0] - s[1] for s in SEL], [s[2] - s[0] for s in SEL]],
                    fmt="o-", color="k", ms=3, capsize=2, lw=0.8)
    axs[3].set_ylim(0, 1.6); axs[3].set_ylabel(r"selection [$\mu$s/element]")
    for a, t in zip(axs, "abcd"):
        a.set_xticks(x); a.set_xticklabels(["4", "8", "16"]); a.set_xlabel("$N$")
        a.text(0.02, 1.03, "(%s)" % t, transform=a.transAxes, weight="bold")
    fig.tight_layout(); save(fig, "fig5_nscale")

# ------------------------------------------------------------------ fig 8
WS = [0, 0.25, 0.5, 1, 2, 4, 8]
def fig8():
    S = setup(); el, dt, rho = S["el"], S["dt"], S["rho"]; rows = []; shown = False
    for ri, (lab, e_f, seed) in enumerate(cc_runs(4)):
        pl, ks, R, x0 = prep(S, e_f, seed); kf = pl.k_fail
        kd = first_flag(ks, R, rho, S["rc"].persistence, el.index[e_f]); miss = not (0 <= kd <= kf)
        for ws in WS:
            H = S["run"](copy.deepcopy(pl), el, mk(S, w_safety=ws), "prop", ks, R, rho, x0=x0)
            ck = H.get("crit_kept")
            if not shown: print("crit_kept repr:", repr(ck)[:120]); shown = True
            ck = -9 if ck is None else int(np.max(np.asarray(ck)))
            rows.append((ws, ri, S["safety_stats"](H)["nviol"] > 0, miss, ck,
                         S["recovery_time"](H, kf, dt), len(H["switch"])))
        print("  fig8 %d/30" % (ri + 1), flush=True)
    A = np.array(rows, dtype=float)
    np.savez("results/fig8_v12.npz", ws=A[:, 0], run=A[:, 1], viol=A[:, 2], miss=A[:, 3],
             crit_kept=A[:, 4], rec=A[:, 5], nsw=A[:, 6])
    x = np.arange(len(WS)); rng = np.random.default_rng(0)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0))
    vm, vs, ret, rec, nsw = [], [], [], [], []
    for ws in WS:
        B = A[A[:, 0] == ws]; v = B[:, 2] > 0; m = B[:, 3] > 0
        vm.append(int((v & m).sum())); vs.append(int((v & ~m).sum()))
        dd = B[B[:, 4] >= 0]; ret.append((int((dd[:, 4] == 1).sum()), len(dd)))
        r = B[:, 5][np.isfinite(B[:, 5])]
        bs = [rng.choice(r, len(r)).mean() for _ in range(2000)]
        rec.append((r.mean(), *np.percentile(bs, [2.5, 97.5]))); nsw.append(B[:, 6].mean())
    print("fig8 violations per ws:", [a + b for a, b in zip(vm, vs)], " (expect 13 at ws=0, 3 at ws>=1)")
    print("fig8 miss/struct:", list(zip(vm, vs)), " retention (kept/detected):", ret)
    print("fig8 recovery mean [CI]:", [tuple(round(q, 3) for q in r) for r in rec], " switches:", nsw)
    tot = [wilson(a + b, 30) for a, b in zip(vm, vs)]
    axs[0].bar(x, [100 * a / 30 for a in vm], 0.6, color="0.8", ec="k", lw=0.5, label="precursor missed")
    axs[0].bar(x, [100 * b / 30 for b in vs], 0.6, bottom=[100 * a / 30 for a in vm], color="0.3",
               ec="k", lw=0.5, label="critical edge lost")
    axs[0].errorbar(x, [q[0] for q in tot], yerr=[[q[0] - q[1] for q in tot], [q[2] - q[0] for q in tot]],
                    fmt="none", color="k", capsize=2, lw=0.6)
    axs[0].set_ylabel("runs with violation [%]"); axs[0].legend(fontsize=6, frameon=False)
    Wr = [wilson(a, b) if b else (np.nan,) * 3 for a, b in ret]
    axs[1].errorbar(x, [q[0] for q in Wr], yerr=[[q[0] - q[1] for q in Wr], [q[2] - q[0] for q in Wr]],
                    fmt="s-", color="k", ms=3, capsize=2, lw=0.8)
    axs[1].set_ylim(0, 105); axs[1].set_ylabel("critical edge kept\nat failure [%]")
    axs[2].errorbar(x, [r[0] for r in rec], yerr=[[r[0] - r[1] for r in rec], [r[2] - r[0] for r in rec]],
                    fmt="o-", color="k", ms=3, capsize=2, lw=0.8, label="recovery (left)")
    axs[2].set_ylim(0, 4); axs[2].set_ylabel("recovery time [s]")
    a2 = axs[2].twinx(); a2.plot(x, nsw, "s--", color="0.5", ms=3, lw=0.8, label="switches (right)")
    a2.set_ylim(0, 4); a2.set_ylabel("structure switches", color="0.4")
    for a, t in zip(axs, "abc"):
        a.set_xticks(x); a.set_xticklabels([str(w) for w in WS]); a.set_xlabel(r"$w_{\mathrm{s}}$")
        a.text(0.02, 1.03, "(%s)" % t, transform=a.transAxes, weight="bold")
    fig.tight_layout(); save(fig, "fig8_wsafety")

for name, fn in (("fig1", fig1), ("fig5", fig5), ("fig2", fig2), ("fig8", fig8)):
    if args.which in ("all", name): fn()
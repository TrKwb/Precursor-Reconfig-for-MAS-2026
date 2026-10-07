# -*- coding: utf-8 -*-
"""Figure generation for the IJRNC manuscript (PDF output) -- shared figures.

This script draws only the figures that have no dedicated script:

    file                 paper     source
    fig1_concept.pdf     Fig. 1    delegated to fig1_concept.py (run once)
    fig2_timeseries.pdf  Fig. 2    results/smoke_*.npz          (exp0_smoke.py)
    fig3_safety.pdf      Fig. 3    results/exp1.npz             (exp1_main.py)
    fig6_mechanism.pdf   Fig. 7    results/exp5_mechanism.npy   (exp5_mechanism.py)

The remaining figures are drawn by their own scripts, each from the result
file that also produces the corresponding table, with its SHA-256 recorded
next to the figure.  They are deliberately NOT drawn here, so that running
this script can never overwrite them with output from superseded files:

    fig4_scaling.pdf     Fig. 6    fig4_scaling.py    <- results/exp2b_seltime.npz
    fig5_nscale.pdf      Fig. 5    fig5_nscale.py     <- results/exp4_campaign2.npz
    fig7_conformal.pdf   Fig. 8    fig7_conformal.py  <- results/exp3b_conformal.npz
    fig8_wsafety.pdf     Fig. 4    fig8_wsafety.py    <- results/exp8_wsafety.csv

Missing inputs are skipped with a notice.  Log axes carry explicit major ticks
and no minor tick labels.  No CJK glyphs.

Usage:  python experiments/figures.py
"""
import sys
import os
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
    "savefig.dpi": 400, "axes.linewidth": 0.7, "lines.linewidth": 1.1,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})
os.makedirs("figs", exist_ok=True)
MADE, SKIP = [], []


def save(fig, name):
    fig.savefig("figs/%s.pdf" % name, bbox_inches="tight", facecolor="white")
    fig.savefig("figs/%s.png" % name, bbox_inches="tight", facecolor="white",
                dpi=300)
    plt.close(fig)
    MADE.append(name)


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=7, length=2, width=0.6)


def panel(ax, letter, pad=4):
    ax.set_title("(%s)" % letter, fontsize=8, fontweight="bold", loc="left",
                 pad=pad)


# ===================== Fig. 1 : architecture =========================
_f1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_concept.py")
if os.path.exists("figs/fig1_concept.pdf"):
    MADE.append("fig1_concept (existing file kept)")
elif os.path.exists(_f1):
    if subprocess.run([sys.executable, _f1]).returncode == 0:
        MADE.append("fig1_concept")
    else:
        SKIP.append(("fig1_concept", "fig1_concept.py failed"))
else:
    SKIP.append(("fig1_concept", "experiments/fig1_concept.py not found"))


# ===================== Fig. 2 : single-trial histories ===============
if os.path.exists("results/smoke_risk.npz"):
    d = np.load("results/smoke_risk.npz")
    ks, R, rho = d["ks"], d["R"], float(d["rho"])
    kf, dt, jf = int(d["k_fail"]), float(d["dt"]), int(d["jf"])
    Hs = {m: np.load("results/smoke_%s.npz" % m) for m in ("fixed", "post", "prop")
          if os.path.exists("results/smoke_%s.npz" % m)}
    t_r, t_f = ks * dt, kf * dt
    fl = np.where(R[:, jf] > rho)[0]
    t_d = t_r[fl[0]] if len(fl) else np.nan

    fig = plt.figure(figsize=(3.35, 4.2))
    gs = GridSpec(3, 1, height_ratios=[1.15, 0.55, 1.30], hspace=0.34,
                  left=0.18, right=0.985, top=0.93, bottom=0.10)
    a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(3))
    for a in (a1, a2, a3):
        a.set_xlim(0, t_r[-1]); tidy(a)
        if t_d == t_d:
            a.axvline(t_d, color="0.4", lw=0.7, ls=(0, (2, 1.5)), zorder=1)
        a.axvline(t_f, color="0.4", lw=0.7, ls=(0, (4, 1.5)), zorder=1)

    # (a) conformal limits.  Legend sits above the axes, right-aligned, so it
    # can overprint neither the rho line nor the lead-time annotation.
    oth = [j for j in range(R.shape[1]) if j != jf]
    a1.plot(t_r, R[:, oth], color="0.78", lw=0.5, zorder=2)
    a1.plot(t_r, R[:, jf], color="k", lw=1.3, zorder=4, label="failing element")
    a1.axhline(rho, color="k", lw=0.8, ls=(0, (4, 2)), zorder=3,
               label=r"$\rho=1-\alpha_e$")
    a1.set_ylim(-0.03, 1.12); a1.set_yticks([0, 0.5, 1.0]); a1.set_xticklabels([])
    a1.set_ylabel(r"$\bar r_e$", fontsize=8)
    if t_d == t_d:
        a1.annotate("", xy=(t_d, 1.0), xytext=(t_f, 1.0),
                    arrowprops=dict(arrowstyle="<|-|>", lw=0.8, color="k",
                                    mutation_scale=6, shrinkA=0, shrinkB=0))
        a1.text((t_d + t_f) / 2, 1.05, "lead %.1f s" % (t_f - t_d),
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    a1.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, handlelength=1.6, columnspacing=1.0, borderaxespad=0.1)
    panel(a1, "a")

    # (b) actuated agents (the two methods coincide by design; see caption)
    for m, col, st, lw, lab in (("post", "0.45", (0, (4, 2)), 1.0, "post-failure"),
                                ("prop", "k", "-", 1.4, "proposed")):
        if m in Hs:
            a2.step(Hs[m]["t"], Hs[m]["nact"], where="post",
                    color=col, ls=st, lw=lw, label=lab)
    if Hs:
        lo = min(Hs[m]["nact"].min() for m in Hs) - 0.6
        hi = max(Hs[m]["nact"].max() for m in Hs) + 0.6
        a2.set_ylim(lo, hi)
        a2.set_yticks(np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1))
    a2.set_xticklabels([]); a2.set_ylabel("actuated\nagents", fontsize=7.5)
    a2.legend(fontsize=6, loc="lower left", handlelength=1.8)
    panel(a2, "b")

    # (c) minimum gap margin
    for m, col, st, lw, lab in (("fixed", "0.68", (0, (1, 1.2)), 0.9, "fixed"),
                                ("post", "0.35", (0, (4, 2)), 1.1, "post-failure"),
                                ("prop", "k", "-", 1.4, "proposed")):
        if m in Hs:
            a3.plot(Hs[m]["t"], Hs[m]["h"].min(axis=1),
                    color=col, ls=st, lw=lw, label=lab)
    ymin = min(Hs[m]["h"].min() for m in Hs) if Hs else -1.0
    a3.axhspan(min(ymin * 1.15, -0.05), 0.0, color="0.9", lw=0, zorder=0)
    a3.axhline(0.0, color="k", lw=1.0, ls=(0, (6, 1.5, 1, 1.5)))
    a3.set_ylim(min(ymin * 1.15, -0.05), 1.05)
    a3.set_ylabel(r"$\min_i h_i$  [m]", fontsize=8)
    a3.set_xlabel(r"time $t$  [s]", fontsize=8)
    a3.legend(fontsize=6, loc="lower right", handlelength=2.0)
    panel(a3, "c")
    save(fig, "fig2_timeseries")
else:
    SKIP.append(("fig2_timeseries", "results/smoke_risk.npz  (run exp0_smoke.py)"))


# ===================== Fig. 3 : safety comparison ====================
if os.path.exists("results/exp1.npz"):
    d = np.load("results/exp1.npz")
    M = ("fixed", "post", "prop")
    LAB = ("fixed", "post-\nfailure", "proposed")
    GREY = ["0.78", "0.52", "0.18"]
    av = {m: d["%s_anyviol" % m].astype(bool) for m in M}
    ar = {m: d["%s_viol_area" % m] for m in M}
    b01 = int(np.sum(~av["post"] & av["prop"]))          # prop only
    b10 = int(np.sum(av["post"] & ~av["prop"]))          # post only
    both = int(np.sum(av["post"] & av["prop"]))
    none = int(np.sum(~av["post"] & ~av["prop"]))

    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.2))
    fig.subplots_adjust(wspace=0.42, left=0.075, right=0.99,
                        top=0.84, bottom=0.23)

    # (a) violation rate over all runs
    v = [100 * av[m].mean() for m in M]
    ax[0].bar(np.arange(3), v, 0.58, color=GREY, edgecolor="k", lw=0.7)
    for i, z in enumerate(v):
        ax[0].text(i, z + max(v) * 0.035, "%.0f%%" % z, ha="center",
                   fontsize=7.5, fontweight="bold")
    ax[0].set_xticks(np.arange(3)); ax[0].set_xticklabels(LAB, fontsize=7.5)
    ax[0].set_ylabel("runs with violation [%]", fontsize=8)
    ax[0].set_ylim(0, max(max(v) * 1.32, 1))

    # (b) violation area, conditional on a violation.  Unconditional boxes
    # collapse onto 0; with few runs a strip plot replaces the box.
    NMIN = 10
    for i, m in enumerate(M):
        y = ar[m][av[m]]
        if y.size == 0:
            continue
        if y.size >= NMIN:
            bp = ax[1].boxplot([y], positions=[i], widths=0.55,
                               showfliers=True, patch_artist=True,
                               flierprops=dict(marker="o", ms=1.8, mfc="0.30",
                                               mec="none", alpha=0.6),
                               medianprops=dict(color="k", lw=1.2),
                               boxprops=dict(lw=0.7), whiskerprops=dict(lw=0.7),
                               capprops=dict(lw=0.7))
            bp["boxes"][0].set_facecolor(GREY[i])
        else:
            rng = np.random.default_rng(0)          # deterministic jitter
            ax[1].plot(i + rng.uniform(-0.12, 0.12, y.size), y, "o",
                       ms=2.8, mfc=GREY[i], mec="k", mew=0.4, lw=0)
    ax[1].set_yscale("log")            # every plotted value is strictly > 0
    ax[1].set_xlim(-0.6, 2.6)
    ax[1].set_xticks(np.arange(3)); ax[1].set_xticklabels(LAB, fontsize=7.5)
    ax[1].set_ylabel(r"violation area [m$\cdot$step]", fontsize=8)
    for i, m in enumerate(M):          # n above the axes, clear of the data
        ax[1].text(i, 1.01, "$n=%d$" % int(av[m].sum()),
                   transform=ax[1].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=6.3, color="0.30")

    # (c) paired contingency table: rows = post-failure, columns = proposed
    tab = [[both, b10], [b01, none]]
    for i in range(2):
        for j in range(2):
            hot = (i == 0 and j == 1)                    # post-only cell
            ax[2].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                          fc="0.78" if hot else "0.96",
                                          ec="k", lw=0.7))
            ax[2].text(j, i - 0.12, "%d" % tab[i][j], ha="center", va="center",
                       fontsize=11 if hot else 9,
                       fontweight="bold" if hot else "normal")
    ax[2].text(1, 0.20, "post only", ha="center", va="center",
               fontsize=5.8, color="0.25")
    ax[2].text(0, 1.20, "prop only", ha="center", va="center",
               fontsize=5.8, color="0.25")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["violation", "none"], fontsize=7)
    ax[2].set_yticks([0, 1]); ax[2].set_yticklabels(["violation", "none"], fontsize=7)
    ax[2].set_xlabel("proposed", fontsize=8)
    ax[2].set_ylabel("post-failure", fontsize=8)
    ax[2].set_xlim(-0.5, 1.5); ax[2].set_ylim(1.5, -0.5)
    for sp in ax[2].spines.values():
        sp.set_visible(False)
    ax[2].tick_params(length=0, labelsize=7)

    for a in ax[:2]:
        tidy(a)
    for i, a in enumerate(ax):
        panel(a, "abc"[i], pad=11 if i == 1 else 4)
    save(fig, "fig3_safety")
else:
    SKIP.append(("fig3_safety", "results/exp1.npz  (run exp1_main.py)"))


# ===================== Fig. 7 (file fig6) : degradation mechanisms ===
if os.path.exists("results/exp5_mechanism.npy"):
    r = np.load("results/exp5_mechanism.npy")
    NAMES = ["AR(1)\n(baseline)", "bias\nramp",
             "emerging\noscillation", "progressive\nquantisation"]
    SHORT = ["AR(1)", "bias", "osc.", "quant."]
    FEAT = ["PE", "DET", "LAM", r"$\log\,$sd", "AC1"]
    n = r.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(6.7, 2.3))
    fig.subplots_adjust(wspace=0.34, left=0.08, right=0.96, top=0.86, bottom=0.28)
    xs = np.arange(n); det = 100 * r[:, 1]
    ax[0].bar(xs, det, 0.55, color=["0.15"] + ["0.62"] * (n - 1),
              edgecolor="k", lw=0.7)
    for i, v in enumerate(det):
        ax[0].text(i, v + 4, "%.0f%%" % v, ha="center",
                   fontsize=7.5, fontweight="bold")
    ax[0].set_ylabel("detection rate [%]", fontsize=8)
    ax[0].set_ylim(0, 122); ax[0].set_yticks([0, 20, 40, 60, 80, 100])
    ax[0].set_xticks(xs); ax[0].set_xticklabels(NAMES[:n], fontsize=6.5)
    tidy(ax[0]); panel(ax[0], "a")
    Fsh = r[:, 4:9]; vmax = max(np.abs(Fsh).max(), 1e-6)
    im = ax[1].imshow(Fsh, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(FEAT, fontsize=7.5)
    ax[1].set_yticks(xs); ax[1].set_yticklabels(SHORT[:n], fontsize=7.5)
    for i in range(n):
        for j in range(5):
            v = Fsh[i, j]
            ax[1].text(j, i, "%+.2f" % v, ha="center", va="center", fontsize=6.5,
                       color="white" if abs(v) > vmax * 0.55 else "k")
    ax[1].set_title("(b) mean feature shift (healthy $\\to$ degraded)",
                    fontsize=7.5, loc="left")
    ax[1].tick_params(length=0)
    cb = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=6); cb.outline.set_linewidth(0.6)
    save(fig, "fig6_mechanism")
else:
    SKIP.append(("fig6_mechanism", "results/exp5_mechanism.npy"))


# ===================== report =========================================
print("generated %d figure(s):" % len(MADE))
for m in MADE:
    print("   figs/%s" % (m if " " in m else m + ".pdf"))
if SKIP:
    print("\nskipped %d (missing input):" % len(SKIP))
    for n_, why in SKIP:
        print("   %-20s <- %s" % (n_, why))

DELEGATED = (("fig4_scaling", "Fig. 6", "fig4_scaling.py"),
             ("fig5_nscale", "Fig. 5", "fig5_nscale.py"),
             ("fig7_conformal", "Fig. 8", "fig7_conformal.py"),
             ("fig8_wsafety", "Fig. 4", "fig8_wsafety.py"))
print("\nnot drawn here (dedicated scripts):")
for f, paper, script in DELEGATED:
    p = "figs/%s.pdf" % f
    src = "figs/%s.source.txt" % f
    if os.path.exists(p):
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        print("   %-20s %-7s %s  %s" % (f, paper, stamp,
              "ok (source recorded)" if os.path.exists(src)
              else "!! no .source.txt -- rerun experiments/%s" % script))
    else:
        print("   %-20s %-7s missing -- run experiments/%s" % (f, paper, script))
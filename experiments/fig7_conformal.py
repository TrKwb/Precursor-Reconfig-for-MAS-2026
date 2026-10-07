# -*- coding: utf-8 -*-
"""Figure 8 (file fig7_conformal.pdf): validity and power of the detector.

The reference for Proposition 1 is NOT a confidence interval on the test
evaluations: with a single calibration set of size n, the realised FAR of a
threshold at rank l = floor(alpha_e (n+1)) is distributed as
Beta(l, n+1-l) over calibration draws.  That spread (sd ~0.0034 at
alpha_e = 0.0143, n = 1200) dominates the test-sampling error, so each panel
shows the 90 % calibration-conditional band.  Clopper-Pearson bars are drawn
thin and cover test sampling only (they also ignore within-run correlation).

(a) raw per-evaluation FAR (the event Proposition 1 bounds);
(b) FAR after the n_p persistence rule (what the selector acts on);
(c) per-run detection.  Ticks sit at the swept levels only.

Usage:  python experiments/fig7_conformal.py
"""
import sys
import os
import math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
from _util import clopper_pearson, sha256_file, _betaq, betainc

NCAL = {64: 1200, 96: 1200}      # <-- set from the calibration probe
OP = 0.20
plt.rcParams.update({"font.size": 7.5, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.7,
                     "legend.frameon": False, "pdf.fonttype": 42})
SRC = "results/exp3b_conformal.npz"
d = np.load(SRC)
WS, AL, nC = [int(w) for w in d["W"]], d["alpha"], int(d["nC"])
ae = AL / nC


def band(a_e, n, lo=0.05, hi=0.95):
    l = math.floor(a_e * (n + 1))
    if l < 1:
        return 0, 0.0, 0.0
    return l, _betaq(lo, l, n + 1 - l), _betaq(hi, l, n + 1 - l)


ST = [dict(marker="o", mfc="k", mec="k", color="k"),
      dict(marker="s", mfc="white", mec="0.3", color="0.3")]
OFF = (0.93, 1.07)
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))
fig.subplots_adjust(wspace=0.45, left=0.075, right=0.985, top=0.86, bottom=0.2)

allb = [band(a, NCAL[W]) for W in WS for a in ae]
ylo = min(b[1] for b in allb if b[0] > 0) / 1.4
yhi = max(b[2] for b in allb) * 1.4
xl = (ae.min() / 1.8, ae.max() * 1.8)
print("W  alpha   l   band90             raw     pct   | flag")
for pi, (kk, nn) in enumerate(((d["raw_k"], d["raw_n"]), (d["flag_k"], d["flag_n"]))):
    a = ax[pi]
    a.plot(xl, xl, color="0.3", lw=0.8, zorder=1)
    a.axvline(OP / nC, color="k", ls=":", lw=0.7, zorder=1)
    for wi, W in enumerate(WS):
        xs = ae * OFF[wi]
        r = kk[wi] / nn[wi]
        if pi == 0:
            for x, a_e, rv in zip(xs, ae, r):
                l, b0, b1 = band(a_e, NCAL[W])
                a.plot([x, x], [b0, b1], color="0.82", lw=5.5, solid_capstyle="butt", zorder=2)
                print("%d %.2f %4d [%.4f,%.4f]  %.4f  %.2f  | %.4f" % (
                    W, a_e * nC, l, b0, b1, rv, betainc(l, NCAL[W] + 1 - l, rv),
                    d["flag_k"][wi][list(ae).index(a_e)] / d["flag_n"][wi][list(ae).index(a_e)]))
        ci = np.array([clopper_pearson(k, n) for k, n in zip(kk[wi], nn[wi])])
        rr = np.maximum(r, ylo)
        a.errorbar(xs, rr, yerr=[rr - np.maximum(ci[:, 0], ylo), np.maximum(ci[:, 1], ylo) - rr],
                   fmt="none", ecolor=ST[wi]["color"], elinewidth=0.6, capsize=1.2, zorder=3)
        a.plot(xs, rr, ls="none", ms=4.2, label="$W=%d$" % W, zorder=4, **ST[wi])
    if pi == 0:
        a.plot([], [], color="0.82", lw=5.5, label="90 % calibration band")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(*xl); a.set_ylim(ylo, yhi)
    a.xaxis.set_major_locator(FixedLocator(ae)); a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["%.4f" % v for v in ae], fontsize=6.2)
    yt = [v for v in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05) if ylo <= v <= yhi]
    a.yaxis.set_major_locator(FixedLocator(yt)); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_yticklabels(["%g" % v for v in yt], fontsize=6.3)
    a.set_xlabel(r"nominal $\alpha_e=\alpha/|\mathcal{C}|$")
    a.set_ylabel("FAR, raw per evaluation" if pi == 0 else "FAR after $n_p$ rule", fontsize=7)
#a.legend(loc="upper left", fontsize=5.8, handletextpad=0.3)
    a.legend(loc="lower right", fontsize=5.8, handletextpad=0.3)
a = ax[2]
for wi, W in enumerate(WS):
    k, n = d["det_k"][wi], d["det_n"][wi]
    v = 100.0 * k / n
    ci = 100.0 * np.array([clopper_pearson(x, y) for x, y in zip(k, n)])
    a.errorbar(AL * (0.96, 1.04)[wi], v, yerr=[v - ci[:, 0], ci[:, 1] - v], ls="-", lw=1.0,
               ms=4.2, elinewidth=0.7, capsize=1.6, label="$W=%d$" % W, **ST[wi])
a.set_xscale("log")
a.xaxis.set_major_locator(FixedLocator(AL)); a.xaxis.set_minor_locator(NullLocator())
a.set_xticklabels(["%g" % v for v in AL]); a.axvline(OP, color="k", ls=":", lw=0.7)
a.set_xlim(AL.min() / 1.4, AL.max() * 1.4); a.set_ylim(0, 108)
a.set_xlabel(r"family-wise level $\alpha$"); a.set_ylabel("detection per run [%]")
a.legend(loc="lower right", fontsize=6.3)
for i, a in enumerate(ax):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.set_title("(%s)" % "abc"[i], loc="left", fontsize=7.5, fontweight="bold")
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig7_conformal.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("figs/fig7_conformal.png", bbox_inches="tight", facecolor="white", dpi=300)
with open("figs/fig7_conformal.source.txt", "w", encoding="utf-8") as f:
    f.write("%s  %s\nNCAL=%s\n" % (sha256_file(SRC), SRC, NCAL))
print("n_eval per point:", d["raw_n"].tolist(), " runs per point:", d["det_n"].tolist())
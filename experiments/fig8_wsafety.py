# -*- coding: utf-8 -*-
"""Figure 5 : scaling with N -- drawn ONLY from results/exp4_campaign2.npz,
the same file that produces Table 2.  SHA-256 of that file is written to
figs/fig5_nscale.source.txt.

Usage:  python experiments/fig5_nscale.py
"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _util import wilson, clopper_pearson, sha256_file

plt.rcParams.update({"font.size": 7.5, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
                     "axes.linewidth": 0.7, "legend.frameon": False,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.5})
SRC = "results/exp4_campaign2.npz"
os.makedirs("figs", exist_ok=True)
d = np.load(SRC)
NS = [int(n) for n in d["NS"]]
x = np.arange(len(NS))
g = lambda N, k: d["N%d_%s" % (N, k)]

fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.2))
fig.subplots_adjust(wspace=0.95, left=0.065, right=0.985, top=0.80, bottom=0.22)

# (a) violation rate; percentage labels above the 100 % baseline bars
W = 0.26
sty = {"fixed": dict(fc="0.88", hatch="////"), "post": dict(fc="0.60", hatch="\\\\\\\\"),
       "prop": dict(fc="0.15")}
lab = {"fixed": "fixed", "post": "post-failure", "prop": "proposed"}
for mi, m in enumerate(("fixed", "post", "prop")):
    k = np.array([g(N, m + "_viol").sum() for N in NS])
    n = np.array([g(N, m + "_viol").size for N in NS])
    v = 100.0 * k / n
    xe = x + (mi - 1) * W
    ax[0].bar(xe, v, W, ec="k", lw=0.6, **sty[m])
    if m == "prop":
        ci = 100.0 * np.array([wilson(a, b) for a, b in zip(k, n)])
        ax[0].errorbar(xe, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none",
                       ecolor="0.55", elinewidth=0.8, capsize=1.8)
        for xi, vi in zip(x, v):
            ax[0].text(xi, 104, "%.0f%%" % vi, ha="center", va="bottom",
                       fontsize=6.5, fontweight="bold")
ax[0].set_ylim(0, 122); ax[0].set_yticks([0, 25, 50, 75, 100])
ax[0].set_ylabel("runs with violation [%]")
fig.legend([Patch(ec="k", lw=0.6, **sty[m]) for m in ("fixed", "post", "prop")],
           [lab[m] for m in ("fixed", "post", "prop")],
           loc="lower left", bbox_to_anchor=(0.055, 0.90), ncol=3, fontsize=6.3,
           handlelength=1.4, columnspacing=1.0)

# (b) per-run detection (left) and raw FAR / alpha_e (right)
k = np.array([g(N, "det").sum() for N in NS]); n = np.array([g(N, "det").size for N in NS])
v = 100.0 * k / n
ci = 100.0 * np.array([clopper_pearson(a, b) for a, b in zip(k, n)])
ax[1].errorbar(x, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="o-", color="k",
               ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.8)
ax[1].set_ylim(0, 108); ax[1].set_ylabel("detection per run [%]")
tw = ax[1].twinx()
r = [g(N, "far_raw_k") / g(N, "far_raw_n") / (g(N, "alpha") / g(N, "nC")) for N in NS]
tw.plot(x, r, "s--", color="0.5", ms=3, lw=0.9)
tw.axhline(1.0, color="0.65", lw=0.6, ls=":")
tw.set_ylim(0, 1.6); tw.set_ylabel(r"FAR$/\alpha_e$", color="0.4", labelpad=1)
tw.tick_params(colors="0.4", labelsize=6.5, length=2)
tw.spines["top"].set_visible(False)

# (c) iso-alpha_e family-wise level
aE4 = float(g(NS[0], "alpha") / g(NS[0], "nC"))
iso = np.array([aE4 * float(g(N, "nC")) for N in NS])
ax[2].bar(x, iso, 0.55, fc="0.45", ec="k", lw=0.6)
ax[2].axhline(1.0, color="k", ls="--", lw=0.8)
ax[2].text(-0.45, 1.05, "no guarantee", fontsize=6, va="bottom")
for xi, vi in zip(x, iso):
    ax[2].text(xi, vi + 0.05, "%.2f" % vi, ha="center", fontsize=6.3, fontweight="bold")
ax[2].set_ylim(0, max(1.6, iso.max() * 1.2))
ax[2].set_ylabel(r"iso-$\alpha_e$ level")

# (d) selection cost per element
q = np.array([np.percentile(1e3 * g(N, "sel_ms") / float(g(N, "nC")), [5, 50, 95]) for N in NS])
ax[3].errorbar(x, q[:, 1], yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]], fmt="^-",
               color="k", ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.8)
for xi, vi, hi in zip(x, q[:, 1], q[:, 2]):
    ax[3].text(xi, hi * 1.04, "%.2f" % vi, ha="center", va="bottom", fontsize=6.3)
ax[3].set_ylim(0, q[:, 2].max() * 1.35); ax[3].set_ylabel(r"selection [$\mu$s/element]")

XL = ["$N=%d$" % N for N in NS]
XL_det = ["$N=%d$\n%d/%d" % (N, a, b) for N, a, b in zip(NS, k, n)]
for i, a in enumerate(ax):
    a.set_xticks(x); a.set_xticklabels(XL_det if i == 1 else XL, fontsize=6.3)
    a.set_xlim(-0.6, len(NS) - 0.4)
    a.spines["top"].set_visible(False)
    if i != 1:
        a.spines["right"].set_visible(False)
    a.tick_params(labelsize=6.5, length=2)
    a.set_title("(%s)" % "abcd"[i], loc="left", fontsize=7.5, fontweight="bold")
if not bool(d["verified"]):
    fig.text(0.5, 0.5, "UNVERIFIED (--dry)", ha="center", va="center", fontsize=26,
             color="red", alpha=0.25, rotation=12)
fig.savefig("figs/fig5_nscale.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("figs/fig5_nscale.png", bbox_inches="tight", facecolor="white", dpi=300)
with open("figs/fig5_nscale.source.txt", "w", encoding="utf-8") as f:
    f.write("%s  %s\n" % (sha256_file(SRC), SRC))
print("figs/fig5_nscale.pdf  <-  %s (sha256 %s)" % (SRC, sha256_file(SRC)[:16]))# -*- coding: utf-8 -*-
"""Figure 4 (file fig8_wsafety.pdf): weight on barrier-critical edges.

Drawn from results/exp8_wsafety.csv (30 collision-capable runs x 7 weights).
(a) violations, stacked by the route through which the tightening failed to
    engage: critical edge lost (crit_kept = 0) or precursor missed
    (crit_kept = 1, i.e. no reconfiguration took place); Wilson 95 % CI on the
    total.  No 'floor' annotation.
(b) critical-edge retention (from the replay) with Wilson 95 % CI.
(c) mean recovery time (left axis) and mean switch count (right axis, twinx),
    each with a 95 % bootstrap interval over runs.

Usage:  python experiments/fig8_wsafety.py
"""
import sys
import os
import csv
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _util import wilson, sha256_file

plt.rcParams.update({"font.size": 7.5, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
                     "axes.linewidth": 0.7, "legend.frameon": False,
                     "pdf.fonttype": 42, "hatch.linewidth": 0.5})
SRC = "results/exp8_wsafety.csv"
with open(SRC, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
WS = sorted({float(r["w_safety"]) for r in rows})
x = np.arange(len(WS))
XL = ["%g" % w for w in WS]


def col(ws, key, typ=float):
    return np.array([typ(r[key]) if r[key] != "" else np.nan
                     for r in rows if float(r["w_safety"]) == ws])


rng = np.random.default_rng(2026)


def boot(v, B=4000):
    v = v[~np.isnan(v)]
    m = rng.choice(v, size=(B, v.size), replace=True).mean(axis=1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


V = [col(w, "violation", int).astype(bool) for w in WS]
K = [col(w, "crit_kept", int).astype(bool) for w in WS]
n = V[0].size
edge = np.array([100.0 * np.sum(v & ~k) / n for v, k in zip(V, K)])
miss = np.array([100.0 * np.sum(v & k) / n for v, k in zip(V, K)])
tot = edge + miss
ci = 100.0 * np.array([wilson(int(v.sum()), n) for v in V])
ret = np.array([100.0 * k.mean() for k in K])
rci = 100.0 * np.array([wilson(int(k.sum()), n) for k in K])

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))
fig.subplots_adjust(wspace=0.55, left=0.07, right=0.93, top=0.86, bottom=0.2)

a = ax[0]
a.bar(x, miss, 0.6, fc="0.80", ec="k", lw=0.6, label="precursor missed")
a.bar(x, edge, 0.6, bottom=miss, fc="0.25", ec="k", lw=0.6, label="critical edge lost")
a.errorbar(x, tot, yerr=[tot - ci[:, 0], ci[:, 1] - tot], fmt="none", ecolor="k",
           elinewidth=0.7, capsize=1.8)
for i in (0, len(WS) - 1):
    a.text(x[i], ci[i, 1] + 2.5, "%.0f%%" % tot[i], ha="center", fontsize=6.5, fontweight="bold")
a.set_ylim(0, 72); a.set_ylabel("runs with violation [%]")
a.legend(fontsize=6, loc="upper right", handlelength=1.2)

a = ax[1]
a.errorbar(x, ret, yerr=[ret - rci[:, 0], rci[:, 1] - ret], fmt="s-", color="k",
           ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.8)
a.axhline(100, color="0.6", ls=":", lw=0.8)
for i in (0, len(WS) - 1):
    a.text(x[i], rci[i, 0] - 3, "%.0f%%" % ret[i], ha="center", va="top", fontsize=6.5, fontweight="bold")

#for i in (0, len(WS) - 1):
#    a.text(x[i], ret[i] - 11, "%.0f%%" % ret[i], ha="center", fontsize=6.5, fon#tweight="bold")

a.set_ylim(0, 108); a.set_ylabel("critical edge retained [%]")

a = ax[2]
R = np.array([boot(col(w, "rec")) for w in WS])
S = np.array([boot(col(w, "nswitch")) for w in WS])
h1 = a.errorbar(x - 0.08, R[:, 0], yerr=[R[:, 0] - R[:, 1], R[:, 2] - R[:, 0]], fmt="o-",
                color="k", ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.8, label="recovery")
a.set_ylim(0, 4); a.set_ylabel("recovery time [s]")
tw = a.twinx()
h2 = tw.errorbar(x + 0.08, S[:, 0], yerr=[S[:, 0] - S[:, 1], S[:, 2] - S[:, 0]], fmt="s--",
                 color="0.5", mfc="white", ms=3.5, lw=1.0, elinewidth=0.7, capsize=1.8,
                 label="switches")
tw.set_ylim(0, 4); tw.set_ylabel("structure switches", color="0.4")
tw.tick_params(colors="0.4"); tw.spines["top"].set_visible(False)
a.legend([h1, h2], ["recovery (left)", "switches (right)"], fontsize=6, loc="lower center")

for i, a in enumerate(ax):
    a.set_xticks(x); a.set_xticklabels(XL, fontsize=6.5)
    a.set_xlabel(r"$w_{\mathrm{s}}$")
    a.spines["top"].set_visible(False)
    if i != 2:
        a.spines["right"].set_visible(False)
    a.set_title("(%s)" % "abc"[i], loc="left", fontsize=7.5, fontweight="bold")
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig8_wsafety.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("figs/fig8_wsafety.png", bbox_inches="tight", facecolor="white", dpi=300)
with open("figs/fig8_wsafety.source.txt", "w", encoding="utf-8") as f:
    f.write("%s  %s\n" % (sha256_file(SRC), SRC))
for w, e, m, t, r, rr, s in zip(WS, edge, miss, tot, ret, R, S):
    print("w_s=%-5g viol %4.1f%% (edge %4.1f, miss %4.1f)  retained %4.1f%%  rec %.3f [%.3f,%.3f]  sw %.2f [%.2f,%.2f]"
          % (w, t, e, m, r, rr[0], rr[1], rr[2], s[0], s[1], s[2]))
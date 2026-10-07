# -*- coding: utf-8 -*-
"""Figure 6 (file fig4_scaling.pdf): selection time vs |C|.
Median with [p5, p95] band; O(|C| log |C|) guide anchored at the LAST measured
median; ticks at every measured size; no minor tick labels.

Usage:  python experiments/fig4_scaling.py
"""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
from _util import sha256_file

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.7,
                     "legend.frameon": False, "pdf.fonttype": 42})
SRC = "results/exp2b_seltime.npz"
d = np.load(SRC)
C = d["nC"].astype(float)
q = np.percentile(d["t_ms"], [5, 50, 95], axis=1)

fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.fill_between(C, q[0], q[2], color="0.85", lw=0, label="[$p_5$, $p_{95}$]")
ax.plot(C, q[1], "o-", color="k", ms=3.5, lw=1.1, label="median")
guide = C * np.log(C); guide *= q[1][-1] / guide[-1]
ax.plot(C, guide, ":", color="0.3", lw=1.0,
        label=r"$\propto|\mathcal{C}|\log|\mathcal{C}|$ (anchored at last point)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_major_locator(FixedLocator(C)); ax.set_xticklabels(["%d" % c for c in C], fontsize=7)
ax.xaxis.set_minor_locator(NullLocator())
lo, hi = q[0].min() / 1.6, q[2].max() * 1.6
cand = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
yt = [v for v in cand if lo <= v <= hi]
ax.set_ylim(lo, hi)
ax.yaxis.set_major_locator(FixedLocator(yt)); ax.set_yticklabels(["%g" % v for v in yt])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"candidate elements $|\mathcal{C}|$"); ax.set_ylabel("selection time [ms]")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(fontsize=6.3, loc="upper left")
os.makedirs("figs", exist_ok=True)
fig.savefig("figs/fig4_scaling.pdf", bbox_inches="tight", facecolor="white")
fig.savefig("figs/fig4_scaling.png", bbox_inches="tight", facecolor="white", dpi=300)
with open("figs/fig4_scaling.source.txt", "w", encoding="utf-8") as f:
    f.write("%s  %s\n%s\n" % (sha256_file(SRC), SRC, str(d["env"])))
for c, a, b, e in zip(C, q[1], q[0], q[2]):
    print("|C|=%4d  median %.4f ms [%.4f, %.4f]  (%.2f %% of 50 ms at p95)" % (c, a, b, e, 2 * e))
print("env:", d["env"])
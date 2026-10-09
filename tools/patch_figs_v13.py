# -*- coding: utf-8 -*-
"""Patch experiments/fig_v12.py for v13 figure fixes."""
import re
FN = "experiments/fig_v12.py"
s = open(FN, encoding="utf-8").read()

NEW_FIG1 = r'''def fig1():
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
'''
s, n = re.subn(r"def fig1\(\):.*?(?=\n# -{10,} fig 2)", lambda m: NEW_FIG1, s, count=1, flags=re.S)
print("fig1 replaced:", n)

REPL = [
 ('axs[0].legend(loc="lower left", fontsize=6.5, frameon=False)',
  'axs[0].legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9)'),
 ('axs[1].legend(loc="upper left", fontsize=6.5, frameon=False)',
  'axs[1].legend(loc="center left", fontsize=6.5, frameon=True, framealpha=0.9)'),
 ('axs[0].set_ylabel("runs with violation [%]"); axs[0].legend(fontsize=6, frameon=False, loc="upper left")',
  'axs[0].set_ylabel("runs with violation [%]"); h_, l_ = axs[0].get_legend_handles_labels(); '
  'fig.legend(h_, l_, fontsize=6.5, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))'),
 ('a.set_xticks(x); a.set_xticklabels(["$N=4$", "$N=8$", "$N=16$"])',
  'a.set_xticks(x); a.set_xticklabels(["4", "8", "16"]); a.set_xlabel("$N$")'),
 ('axs[1].set_ylabel("critical edge kept [%]\\n(detected runs)")',
  'axs[1].set_ylabel("critical edge kept\\nat failure [%]")'),
]
for old, new in REPL:
    c = s.count(old); s = s.replace(old, new); print("%-60s %d" % (old[:60], c))
open(FN, "w", encoding="utf-8").write(s)
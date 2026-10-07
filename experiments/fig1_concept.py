# -*- coding: utf-8 -*-
"""Figure 1 : system architecture (conceptual, no data required).

NOTE on mathtext: matplotlib's mathtext is not full LaTeX.  Avoid \Pr, and avoid
the thin space \, immediately inside brackets -- both break the parser and
surface as a misleading "Unknown symbol: \le".  Every expression here is checked.

Layout note: the inter-block gap is 8 units and the widest arrow label is 7.6,
so the labels clear the block borders.  They also carry a white bbox as a
safeguard.

Usage:  python experiments/fig1_concept.py
Output: figs/fig1_concept.pdf (and .png)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "mathtext.fontset": "dejavuserif", "axes.unicode_minus": False,
    "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
})
os.makedirs("figs", exist_ok=True)

fig = plt.figure(figsize=(6.6, 3.15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc="white", lw=0.9, ls="-", r=1.0, z=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=%.2f" % r,
        fc=fc, ec="black", lw=lw, ls=ls, zorder=z, mutation_aspect=0.55))


def txt(x, y, s, size=7.4, w="normal", c="black", ha="center", va="center",
        bbox=None):
    ax.text(x, y, s, fontsize=size, fontweight=w, color=c,
            ha=ha, va=va, zorder=5, linespacing=1.45, bbox=bbox)


def arrow(x0, y0, x1, y1, lw=1.0, style="-|>", ms=9, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
        lw=lw, color="black", ls=ls, shrinkA=0, shrinkB=0, zorder=4))


# ---------------- environment strip ----------------
box(1, 88.5, 98, 9.5, fc="0.90", lw=0.8)
txt(50, 93.2,
    "environment:  link loss   $\\cdot$   sensor fault   $\\cdot$   "
    "agent dropout   $\\cdot$   communication delay", 7.8)

# ---------------- three stages ----------------
BY, BH = 26.0, 56.0
AX, AW = 1.0, 27.5          # gaps widened to 8.0 units
BX, BW = 36.5, 26.5
CX, CW = 71.0, 28.0
HDR = 7.6

for x, w, ttl in ((AX, AW, "A   precursor risk"),
                  (BX, BW, "B   structure selection"),
                  (CX, CW, "C   safe recovery control")):
    box(x, BY, w, BH)
    ax.add_patch(FancyBboxPatch(
        (x, BY + BH - HDR), w, HDR,
        boxstyle="round,pad=0,rounding_size=1.0",
        fc="0.22", ec="black", lw=0.9, zorder=3, mutation_aspect=0.55))
    txt(x + w / 2, BY + BH - HDR / 2, ttl, 7.8, "bold", "white")

AC, BC, CC = AX + AW / 2, BX + BW / 2, CX + CW / 2

# --- A -------------------------------------------------------------
txt(AC, 70.5, "diagnostic signals $y_e(k)$", 6.9)
txt(AC, 65.0, "RQA / permutation entropy", 7.3, "bold")
txt(AC, 60.2, r"$\downarrow$", 8.5)
txt(AC, 55.4, "split conformal prediction", 7.3, "bold")
txt(AC, 50.6, r"$\mathrm{P}[\bar r_e > \rho_e] \leq \alpha_e$", 7.2)
txt(AC, 45.6, r"$\downarrow$", 8.5)
txt(AC, 40.4, r"$\bar r_e(\alpha_e)$", 8.0, "bold")
txt(AC, 33.5, r"$\mathcal{S}(\alpha,\rho)$", 8.4, "bold")
txt(AC, 29.5, "admissible structures", 6.5)

# --- B -------------------------------------------------------------
txt(BC, 70.5, r"candidates  $E \cup V \cup U$", 6.9)
txt(BC, 65.0, "downward-closed", 7.3, "bold")
txt(BC, 60.6, "independence system", 7.3, "bold")
txt(BC, 55.6, r"$\downarrow$", 8.5)
txt(BC, 50.6, "restricted graphic matroid", 7.1)
txt(BC, 46.0, "Kruskal  (polynomial)", 7.3, "bold")
txt(BC, 41.0, r"$\downarrow$", 8.5)
txt(BC, 33.5, r"$s^{\star} = (E,V,U)$", 8.2, "bold")
txt(BC, 29.5, "selected structure", 6.5)

# --- C -------------------------------------------------------------
txt(CC, 70.5, "distributed MPC  $+$  CBF", 7.3, "bold")
txt(CC, 65.0, r"$\bar r_e \rightarrow$ chance constraint", 7.0)
txt(CC, 60.2, r"$\downarrow$", 8.5)
txt(CC, 55.2, r"$\mathcal{X}_{\mathrm{f}}(s)$,  $\Omega(s)$  invariant sets", 7.0)
txt(CC, 50.4, r"$\downarrow$", 8.5)
txt(CC, 45.4, r"safe transition,  dwell time $\tau_d$", 7.2, "bold")
txt(CC, 38.0, "recursive feasibility $+$ safety", 7.1)
txt(CC, 33.5, r"with probability $1 - \alpha$", 7.8, "bold")

# ---------------- inter-stage arrows ----------------
LBL = dict(boxstyle="square,pad=0.15", fc="white", ec="none")
arrow(AX + AW + 0.8, 50.0, BX - 0.8, 50.0, lw=1.2, ms=11)
txt((AX + AW + BX) / 2, 54.5, r"$\mathcal{S}(\alpha,\rho)$", 7.0, "bold",
    bbox=LBL)
arrow(BX + BW + 0.8, 50.0, CX - 0.8, 50.0, lw=1.2, ms=11)
txt((BX + BW + CX) / 2, 54.5, r"$s^{\star}$", 7.0, "bold", bbox=LBL)

# ---------------- guarantee row ----------------
# Proposition NUMBERS deliberately live in the LaTeX caption, not here.
# They are assigned by LaTeX and have already drifted twice (3/4 in this
# script, 4/5 in the committed PDF, 6 required by the text) because the
# figure and the manuscript are edited independently.  Name the guarantee;
# let \ref{} carry the number.  Do NOT put "Proposition n" back in here.
PY, PH = 16.5, 6.0          # centre unchanged (19.5); box tightened for 1 line
GUARANTEES = (
    (AX, AW, r"coverage  $1 - \alpha$"),
    (BX, BW, r"exact, polynomial time"),
    (CX, CW, r"feasibility across a switch"),
)
for x, w, lab in GUARANTEES:
    box(x, PY, w, PH, fc="0.94", lw=0.8, ls=(0, (3, 1.6)))
    txt(x + w / 2, PY + PH / 2, lab, 7.3, "bold")

# ---------------- feedback path ----------------
FY = 8.5
arrow(CC, PY - 0.5, CC, FY, lw=0.9, style="-")
ax.plot([AC, CC], [FY, FY], lw=0.9, color="black", zorder=4)
arrow(AC, FY, AC, PY - 0.5, lw=0.9, ms=9)
txt(50, FY + 2.9,
    "realised structure and control outcome fed back to the next risk evaluation",
    6.8)

for ext in ("pdf", "png"):
    fig.savefig("figs/fig1_concept.%s" % ext,
                bbox_inches="tight", facecolor="white")
print("  figs/fig1_concept.pdf / .png")
plt.close(fig)
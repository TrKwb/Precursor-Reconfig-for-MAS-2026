# -*- coding: utf-8 -*-
import re, shutil, sys
FN = sys.argv[1]; s = open(FN, encoding="utf-8").read()
shutil.copy(FN, FN.replace(".tex", "_v14_backup.tex"))
rx = lambda old: r"\s+".join(re.escape(t) for t in old.split())
R = [
("Remark 1(i)",
 r"""(G2) holds by Assumption~\ref{asm:nonretreat} for a failed agent as well as a healthy one;""",
 r"""(G2) holds by Assumption~\ref{asm:nonretreat} for a failed agent at all times, and for a healthy agent until a failure is observed;"""),
("Table 1 always-stop",
 r"""safety guarantee & --- & yes & no & Thm.~\ref{thm:lead} & Thm.~\ref{thm:lead}\\""",
 r"""safety guarantee & --- & no$^{a}$ & no & Thm.~\ref{thm:lead} & Thm.~\ref{thm:lead}\\"""),
("Table 1 footnote",
 r"""The three unrecovered runs of the proposed scheme are the missed precursors.""",
 r"""The three unrecovered runs of the proposed scheme are the missed precursors.  $^{a}$\cond{always-stop} uses a valid tube but neither shield nor hard constraint, so its safety here is empirical."""),
("earlier version",
 r"""A fixed window $[k_f-2,k_f]$, used in an earlier version of the check, is not the hypothesis of the theorem and does not exclude this run; the state-based hypothesis does.""",
 r"""A fixed latch window such as $[k_f-2,k_f]$ is not the hypothesis of the theorem and does not exclude this run; the state-based hypothesis does."""),
("P3 sup",
 r"""and runs latched more than $\sup L^\star$ steps before failure do not.""",
 r"""and runs latched more than $\sup L^\star$ steps before failure, from a state in the region over which the supremum is taken, do not."""),
]
miss = []
for name, old, new in R:
    s, n = re.subn(rx(old), lambda m: new, s, count=1)
    print("%-20s %s" % (name, "OK" if n else "NOT FOUND")); miss += [] if n else [name]
open(FN, "w", encoding="utf-8").write(s); print("not found:", miss or "none")
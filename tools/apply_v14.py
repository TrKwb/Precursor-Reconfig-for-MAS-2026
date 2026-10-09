# -*- coding: utf-8 -*-
"""v14 fixes. Usage: python -X utf8 tools/apply_v14.py <path-to-main.tex>"""
import re, shutil, sys
FN = sys.argv[1]
s = open(FN, encoding="utf-8").read()
shutil.copy(FN, FN.replace(".tex", "_v13_backup.tex"))
rx = lambda old: r"\s+".join(re.escape(t) for t in old.split())
R = [
("Assumption 2",
 r"""Each agent $j$ satisfies $v_j\ge0$ and $p_j(k{+}1)\ge p_j(k)$; a dead actuator, for which $u_j\equiv0$ and $c>0$, satisfies this automatically. It is imposed as a linear constraint in each agent's problem.""",
 r"""Until a failure is observed, each agent $j$ satisfies $v_j\ge0$ and $p_j(k{+}1)\ge p_j(k)$, imposed as a linear constraint in its own problem.  A dead actuator, for which $u_j\equiv0$ and $c>0$, satisfies it automatically and for all time.  After a failure the constraint is lifted for the other agents, so that the group can re-form around the stopped agent."""),
("Theorem 1 claim",
 r"""Then $h_i(x(k))\ge0$ for all pairs $i$ and all $k$.""",
 r"""Then $h_{j-1}(x(k))\ge0$ for all $k$, and $h_i(x(k))\ge0$ for all pairs $i$ and all $k\le k_f$."""),
("Theorem 1 proof",
 r"""Pairs whose predecessor does not fail: the envelope contract is valid throughout, and Lemma~\ref{lem:F} gives $h_i\ge0$.""",
 r"""Other pairs up to $k_f$: every predecessor is healthy and non-retreating, so the envelope contract is valid and Lemma~\ref{lem:F} gives $h_i\ge0$.  After $k_f$ the other agents may retreat to re-form the group, and the theorem makes no claim about their pairs."""),
("Corollary 1",
 r"""Let $L$ be the event of Theorem~\ref{thm:lead}. Then every violation occurs in the pair of a failing agent, $\{\text{violation}\}\subseteq L^c$, and \[ \Pr[\text{violation}]\le\Pr[L^c]""",
 r"""Let $L$ be the event of Theorem~\ref{thm:lead}, and let $V$ be the event that the pair of a failing agent violates at any time, or any pair violates before $k_f$.  Then $V\subseteq L^c$, and \[ \Pr[V]\le\Pr[L^c]"""),
("P2 prediction",
 r"""\item[(P2)] violations occur only in the pair of the failing agent;""",
 r"""\item[(P2)] no violation occurs outside $V$, that is, in another pair during the re-formation after failure, which the theorem does not cover;"""),
("P2 result",
 r"""no violation propagated to another pair;""",
 r"""no violation propagated to another pair, including during the re-formation after failure;"""),
("360 abstract", r"""over 360 run-conditions""", r"""over 270 run-conditions"""),
("360 sec6.3",
 r"""Over $360$ run-conditions (four latched variants $\times$ three lead settings $\times$ thirty runs at $N=4$):""",
 r"""Over $270$ run-conditions (three latched variants $\times$ three lead settings $\times$ thirty runs at $N=4$):"""),
("360 conclusion", r"""none of $360$ run-conditions""", r"""none of $270$ run-conditions"""),
("Campaign I Lc",
 r"""On Campaign~I, every detected cc run had a lead of at least $0.40$\,s, longer than $\sup L^\star\Delta t=0.35$\,s, so $\Pr[L^c]$ equals the miss rate there.""",
 r"""On Campaign~I, every detected cc run (minimum lead $0.40$\,s) was latched throughout the last two steps before failure and none of them violated, so the observed violation rate equals the miss rate there."""),
]
miss = []
for name, old, new in R:
    s, n = re.subn(rx(old), lambda m: new, s, count=1)
    print("%-18s %s" % (name, "OK" if n else "NOT FOUND")); miss += [] if n else [name]
open(FN, "w", encoding="utf-8").write(s)
print("not found:", miss or "none")
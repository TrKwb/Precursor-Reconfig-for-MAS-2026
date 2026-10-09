# -*- coding: utf-8 -*-
"""Apply v13 text fixes to main.tex (whitespace-insensitive).  Usage: python -X utf8 tools/apply_v13.py"""
import re, shutil, sys
FN = sys.argv[1] if len(sys.argv) > 1 else "main.tex"
s = open(FN, encoding="utf-8").read()
shutil.copy(FN, "main_v12_backup.tex")
rx = lambda old: r"\s+".join(re.escape(t) for t in old.split())

R = [
("fig1 caption CHECK",
 r"""\CHECK{regenerate the figure: stage C must read ``latched stop contract + shield; safe if reached before failure'', not ``with probability $1-\alpha$''.}""", ""),

("Assumption 4",
 r"""Each agent's own state evolves by~\eqref{eq:dyn} under the applied input. \CHECK{If additive process disturbances act on~\eqref{eq:dyn} in the simulations, the sets of Section~\ref{sec:lead} must be tightened by their reachable set as in~\citep{mayne}.}""",
 r"""Each agent's own state evolves by~\eqref{eq:dyn} under the applied input.  In the simulations there is no process disturbance; the only sources of randomness are the initial-position jitter and the diagnostic signals.  Additive disturbances would require the shield sets of Section~\ref{sec:lead} to be tightened by their reachable set, as in~\citep{mayne}."""),

("shield CHECK",
 r"""otherwise agent $i$ brakes maximally. \CHECK{matches \texttt{policy\_T} in \texttt{runner.py}.}""",
 r"""otherwise agent $i$ brakes maximally.  Shield interventions and hard-tier relaxations are both counted per agent and step."""),

("sec 5.6 sigma",
 r"""Instead the risk of agent $j$ changes its followers' contract. Below the flag threshold it enters through the back-off $\sigma_j$ of Remark~\ref{rem:tubes}(ii); at the flag the contract is latched to stop. \CHECK{$\sigma_j$ depends on $\rb_j$ in the implementation.}""",
 r"""Instead the risk of agent $j$ changes its followers' contract.  In the implementation it does so only through the flag: below the threshold a follower trusts the predecessor's broadcast plan while the shield enforces the envelope contract, and at the flag the contract is latched to stop.  A risk-dependent back-off, as permitted by Remark~\ref{rem:tubes}(ii), is not used."""),

("conditions: disturbance",
 r"""share the initial condition, disturbance realisation and diagnostic signals of each run""",
 r"""share the initial condition and the diagnostic signals of each run (there is no process disturbance)"""),

("tighten-only item",
 r"""\item \cond{tighten-only}: communicated tube with risk-dependent back-off, no latch, structure frozen. Not covered by Theorem~\ref{thm:lead}.""",
 r"""\item \cond{tighten-only}: the constraint tightening of the earlier design with the structure frozen, so that no critical edge is dropped; neither latch nor shield.  Not covered by Theorem~\ref{thm:lead}."""),

("latch-reconf item",
 r"""\item \cond{latch-reconf}: as proposed, with forced selection and pre-emptive structural reconfiguration.""",
 r"""\item \cond{latch-reconf}: \cond{latch-held} (no release) with forced selection and pre-emptive structural reconfiguration."""),

("6.2 tighten-only",
 r"""It is not covered by Theorem~\ref{thm:lead}: its back-off is a heuristic function of $\rb$ and its validity rests on (G2) for a predecessor that may stop.""",
 r"""It has neither latch nor shield, so nothing in it enforces a valid bound on a predecessor that may stop; its safety on these runs is empirical."""),

("P3 run /7",
 r"""The only violation in a latched run occurred at oracle lead $0.4$\,s (\cond{latch-held} and \cond{latch-reconf}, run $(u,2,2)/7$): the latch began five steps before failure with a gap margin of only $0.220$\,m, and $F_{\mathrm{stop}}$ was not reached by $k_f$. This run lies in $L^c$ through the third term of Corollary~\ref{cor:bound}. Under the proposed release rule the same run reached $F_{\mathrm{stop}}$ in time, because its trajectory before the latch differed. \CHECK{reduced margin at latch onset caused by an earlier false-alarm latch on the predecessor's own predecessor.}""",
 r"""The only violation in a latched run occurred at oracle lead $0.4$\,s under \cond{latch-held} and \cond{latch-reconf}, both of which never release a latch (run $(u,2,2)/7$).  A false alarm on the actuator of the leading agent at $t=4.0$\,s, before degradation began, had latched its follower to the stop contract for the rest of the run.  That follower opened its forward gap to a margin of $1.57$\,m, and the agents behind it, still under the envelope contract, closed up to margins of $0.22$--$0.30$\,m: safe against a braking predecessor, with occasional shield interventions, but far from the $0.883$\,m that $F_{\mathrm{stop}}$ requires.  When the next agent was flagged four steps before its own failure, its follower could not reach $F_{\mathrm{stop}}$ in time, and the run lies in $L^c$ through the third term of Corollary~\ref{cor:bound}.  Under the proposed release rule the false latch was released at $t=6.35$\,s, the margins returned to $0.69$\,m, and the same follower reached $F_{\mathrm{stop}}$ in two steps without violation.  A false alarm that is never released therefore costs more than performance: it erodes the margin that a later, genuine flag needs."""),

("6.3 counters CHECK",
 r"""\CHECK{both counters are per agent-step; hard relaxations exceed shield steps because a relaxed QP solution may still lie in $F_\beta$.}""",
 r"""Hard-tier relaxations can exceed shield interventions because a relaxed QP solution may still lie in $F_\beta$, in which case it is applied."""),

("abrupt / recovery",
 r"""The non-retreat contract of Assumption~\ref{asm:nonretreat} forbids the follower to back away from a stopped predecessor once inside $d_{\min}$. Relaxing the contract after a failure has been observed restores recovery in principle, but the variant tested here did not change the outcome. \CHECK{\texttt{nonretreat\_scope=prefail} is wired in \texttt{runner.py}.} This is the price of the contract, and it is reported as a limitation (Section~\ref{sec:lim}).""",
 r"""The cause is not the non-retreat constraint, which is lifted once a failure has been observed, but the shield: outside $F_\beta$ its only action is maximal braking, and since $F_\beta$ requires $h\ge0$, a follower that has entered $d_{\min}$ of a stopped predecessor is held at rest there.  Suspending the shield while $h_i<0$ would not affect Theorem~\ref{thm:lead}, whose event excludes such states, and would let the QP re-form the group; we have not evaluated it (Section~\ref{sec:lim})."""),

("6.4 latch-reconf cost",
 r"""adding forced reconfiguration to the proposed contract (\cond{latch-reconf}) doubles the spacing cost without changing any violation""",
 r"""adding forced reconfiguration to the held latch (\cond{latch-reconf}) doubles the spacing cost relative to \cond{latch-held} ($0.039$ against $0.018$\,m) without changing any violation"""),

("Table 2 note",
 r"""The $26/30$ of \cond{latch-reconf} without a precursor is not a precursor effect: in four runs a false alarm on a critical edge had already placed the follower under the stop fallback.""",
 r"""The $26/30$ of \cond{latch-reconf} without a precursor is not a precursor effect, since none was present, and is not used in any claim."""),

("fig2 caption",
 r"""\CHECK{regenerate with the proposed configuration (\texttt{T\_latch\_rel10}).}""",
 r"""Run $(u,2,2)/0$, lead $1.20$\,s."""),

("fig3 caption",
 r"""(b) Critical-edge retention by an out-of-loop replay of the selection, which may understate retention by up to two runs: from $57$\,\% to $90$\,\%. (c) Recovery time and switch count. \CHECK{recompute panel (c) with the recovery definition of Section~\ref{sec:setup}.}""",
 r"""(b) Critical edge in the structure in force at failure (closed-loop record): from $57$\,\% to $90$\,\%, saturating at $\ws\ge1$.  (c) Mean recovery time (left) and switch count (right) with $95$\,\% bootstrap intervals: $2.945\to2.887$\,s and $3.10\to2.97$."""),

("Table 5 lead",
 r"""mean lead time [s] & $2.36$ & $2.13$ & $1.23$\\""",
 r"""mean lead time [s] & $2.36$ & $2.23$ & $1.32$\\"""),

("fig4 caption CHECK",
 r"""\CHECK{regenerate panel (a) with the proposed scheme.}""", ""),

("limitation (ii)",
 r"""Under the non-retreat contract a follower that has entered $d_{\min}$ of a stopped predecessor cannot back away, so missed precursors and abrupt faults remain unrecovered, whereas the post-failure baseline recovers. A contract that permits retreat after a failure has been observed would remove this cost; it was not effective in the implementation tested here.""",
 r"""The shield has no recovery mode: once a follower is inside $d_{\min}$ of a stopped predecessor, maximal braking holds it there, so missed precursors and abrupt faults remain unrecovered, whereas the post-failure baseline recovers.  A recovery mode that suspends the shield while $h_i<0$ leaves the theorem intact and is the natural fix."""),

("conclusion",
 r"""a contract that permits recovery after a missed precursor;""",
 r"""a recovery mode for the shield after a missed precursor;"""),

("ref Dakos", r"""DOI:10.1371/journal.pone.0041010 \CHECK{}""", r"""DOI:10.1371/journal.pone.0041010"""),
("ref Kerrigan", r"""Conference on Control}; 2000. \CHECK{pages}""", r"""Conference on Control}; 2000."""),
]

miss = []
for name, old, new in R:
    s, n = re.subn(rx(old), lambda m: new, s, count=1)
    print("%-24s %s" % (name, "OK" if n else "NOT FOUND"))
    if not n: miss.append(name)
open(FN, "w", encoding="utf-8").write(s)
print("\nwritten %s (backup main_v12_backup.tex);  not found: %s" % (FN, miss or "none"))
for i, l in enumerate(s.splitlines(), 1):
    if "\\CHECK{" in l or "\\TBD{" in l:
        print("  remaining line %d: %s" % (i, l.strip()[:110]))
# -*- coding: utf-8 -*-
"""Task B : risk-dependent admissible structure set and structure selection.

Safety-critical edges: an edge (i,j) is critical when a barrier function couples
x_i and x_j (consecutive pairs in the 1-D chain).  Remedy 1 subtracts w_safety
from their weight; Remedy 2 (cfg.force_critical) forces a maximal forest of the
admissible critical edges into the solution by pre-merging them in union-find
(matroid contraction, Proposition 5).

Graduated fallback: MASRECON_FALLBACK in {'two','quantile','fixed'}.
"""
import os
import time
import numpy as np
from .structure import phi, mst_edges, connected_on, sensing_covered

FALLBACK = os.environ.get("MASRECON_FALLBACK", "two")


def admissible_elements(elements, rbar, rho, dead=()):
    """A(alpha) : elements that are alive and below the risk threshold."""
    dead = set(dead)
    return [e for e in elements.all
            if (e not in dead) and (rbar[elements.index[e]] <= rho)]


def is_safety_critical(e, N):
    if e[0] != "e":
        return False
    return abs(e[1] - e[2]) == 1


def edge_weights(elements, Ee, rbar, prev_set, cfg, N):
    """w_e = w_risk*rbar_e + w_change*1[new] + w_perf - w_safety*1[critical]"""
    ws = getattr(cfg, "w_safety", 0.0)
    w = {}
    for e in Ee:
        w[e] = (cfg.w_risk * rbar[elements.index[e]]
                + cfg.w_change * (0.0 if e in prev_set else 1.0)
                + cfg.w_perf * 0.15
                - ws * (1.0 if is_safety_critical(e, N) else 0.0))
    return w


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def kruskal_forced(Ee, w, act, N):
    """Kruskal on the induced ground set with the admissible critical edges
    pre-merged (a maximal forest of them).  Returns the edge list."""
    actset = set(act)
    Ei = [e for e in Ee if e[1] in actset and e[2] in actset]
    parent = {i: i for i in act}
    comp = len(act)
    sel = []
    for e in Ei:                                   # forced forest F
        if not is_safety_critical(e, N):
            continue
        a, b = _find(parent, e[1]), _find(parent, e[2])
        if a != b:
            parent[a] = b
            sel.append(e)
            comp -= 1
    for e in sorted(Ei, key=lambda e: w[e]):       # remainder, greedy
        if comp <= 1:
            break
        a, b = _find(parent, e[1]), _find(parent, e[2])
        if a != b:
            parent[a] = b
            sel.append(e)
            comp -= 1
    return sel


def _try_select(elements, A, rbar, prev, cfg, N):
    Ue = [e for e in A if e[0] == "u"]
    Ve = [e for e in A if e[0] == "v"]
    Ee = [e for e in A if e[0] == "e"]
    act = sorted(i for _, i, _ in Ue)
    if len(act) < N - cfg.F_max:                            # (R1)
        return None, False
    prev_set = set(prev) if prev else set()
    w = edge_weights(elements, Ee, rbar, prev_set, cfg, N)
    if getattr(cfg, "force_critical", False):
        Esel = kruskal_forced(Ee, w, act, N)                # Proposition 5
    else:
        Esel = mst_edges(Ee, w, act, N)                     # Proposition 4
    if not connected_on(Esel, act, N):                      # (R2)
        return None, False
    Vsel = list(Ve)
    if not sensing_covered(Esel, Vsel, act, N):             # (R3)
        return None, False
    s = Esel + Vsel + Ue
    return (s, True) if phi(s, N, cfg.F_max) else (None, False)


def _intermediate_set(elements, A, rbar, dead, policy):
    if policy == "two":
        return None
    if policy == "fixed":
        return admissible_elements(elements, rbar, 0.5, dead=dead)
    alive = [e for e in elements.all if e not in set(dead)]
    inA = set(A)
    flagged = [rbar[elements.index[e]] for e in alive if e not in inA]
    if not flagged:
        return None
    thr = float(np.quantile(flagged, 0.25))
    A1 = [e for e in alive if rbar[elements.index[e]] <= thr]
    return A1 if len(A1) > len(A) else None


def select_structure(elements, A, rbar, prev, cfg, N,
                     dead=(), return_stage=False, policy=None):
    """Algorithm 1 with graduated fallback.
    stage 0 = nominal, 1 = intermediate, 2 = all alive, -1 = failed."""
    pol = policy if policy is not None else FALLBACK
    t0 = time.perf_counter()

    s, ok = _try_select(elements, A, rbar, prev, cfg, N)
    stage = 0
    if not ok:
        A1 = _intermediate_set(elements, A, rbar, dead, pol)
        if A1 is not None:
            s, ok = _try_select(elements, A1, rbar, prev, cfg, N)
            stage = 1
    if not ok:
        A2 = admissible_elements(elements, rbar, 2.0, dead=dead)
        s, ok = _try_select(elements, A2, rbar, prev, cfg, N)
        stage = 2
    if not ok:
        stage = -1
    dt = time.perf_counter() - t0
    return (s, ok, dt, stage) if return_stage else (s, ok, dt)
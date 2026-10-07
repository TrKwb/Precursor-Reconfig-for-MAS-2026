# -*- coding: utf-8 -*-
"""Structure s = (E, V, U): candidate elements, requirement Phi(s), and the
graph utilities used by the selector.  Pure Python, no networkx.
"""
from itertools import combinations


class Elements:
    """Candidate element universe C = E_cand + V_cand + U_cand."""

    def __init__(self, N, edge_mode="complete", ring_chords=2):
        self.N = N
        if edge_mode == "complete":
            self.E = [("e", i, j) for i, j in combinations(range(N), 2)]
        else:
            E = [("e", i, (i + 1) % N) for i in range(N)]
            for k in range(2, 2 + ring_chords):
                E += [("e", i, (i + k) % N) for i in range(N)]
            self.E = sorted({("e", min(a, b), max(a, b)) for _, a, b in E})
        self.V = [("v", i, i) for i in range(N)]
        self.U = [("u", i, i) for i in range(N)]
        self.all = self.E + self.V + self.U
        self.index = {e: k for k, e in enumerate(self.all)}

    def __len__(self):
        return len(self.all)


def neighbors(edges, N):
    adj = {i: set() for i in range(N)}
    for _, a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def connected_on(edges, nodes, N):
    """Is the subgraph induced on `nodes` connected, using `edges`?"""
    nodes = list(nodes)
    if len(nodes) <= 1:
        return True
    adj = neighbors(edges, N)
    ns = set(nodes)
    seen, stack = {nodes[0]}, [nodes[0]]
    while stack:
        x = stack.pop()
        for y in adj[x] & ns:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen == ns


def sensing_covered(edges, sensors, nodes, N):
    """Every node in `nodes` has its own sensor or a one-hop neighbour with one."""
    have = {i for _, i, _ in sensors}
    adj = neighbors(edges, N)
    return all((i in have) or bool(adj[i] & have) for i in nodes)


def phi(s, N, F_max=2):
    """Structure requirement predicate.
    (R1) at least N - F_max working actuators
    (R2) the communication graph restricted to actuated agents is connected
    (R3) every actuated agent is sensing-covered
    """
    E = [e for e in s if e[0] == "e"]
    V = [e for e in s if e[0] == "v"]
    U = [e for e in s if e[0] == "u"]
    act = sorted(i for _, i, _ in U)
    if len(act) < N - F_max:
        return False
    if not connected_on(E, act, N):
        return False
    if not sensing_covered(E, V, act, N):
        return False
    return True


def mst_edges(edges, weights, nodes, N):
    """Kruskal on the graphic matroid; returns a spanning forest of `nodes`."""
    parent = {i: i for i in range(N)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    ns = set(nodes)
    for e in sorted(edges, key=lambda e: weights[e]):
        _, a, b = e
        if a not in ns or b not in ns:
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            out.append(e)
    return out

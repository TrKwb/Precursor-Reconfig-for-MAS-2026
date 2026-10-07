# -*- coding: utf-8 -*-
"""Statistics and bookkeeping helpers shared by the experiment scripts.

Standard library + numpy only (no scipy).  Clopper-Pearson, McNemar and Fisher
are exact; Wilson and Cochran-Armitage are the usual asymptotic forms.
"""
import hashlib
import math
import platform
import sys
import time

import numpy as np

Z95 = 1.959963984540054


# ---------------------------------------------------------------- intervals
def wilson(k, n, z=Z95):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _betacf(a, b, x, maxit=50000, eps=1e-15):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def _betaq(q, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else _betaq(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _betaq(1 - alpha / 2, k + 1, n - k)
    return lo, hi


# -------------------------------------------------------------------- tests
def _lcomb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p from the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.exp(_lcomb(n, i) - n * math.log(2.0)) for i in range(k + 1))
    return min(1.0, 2.0 * s)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]]."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    def lp(x):
        return _lcomb(r1, x) + _lcomb(r2, c1 - x) - _lcomb(n, c1)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p0 = lp(a)
    return min(1.0, sum(math.exp(lp(x)) for x in range(lo, hi + 1)
                        if lp(x) <= p0 + 1e-9))


def cochran_armitage(k, n, scores=None):
    """Two-sided Cochran-Armitage trend test; returns (z, p)."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    s = np.arange(1, len(k) + 1, dtype=float) if scores is None else np.asarray(scores, float)
    N, K = n.sum(), k.sum()
    pb = K / N
    T = np.sum(s * (k - n * pb))
    V = pb * (1 - pb) * (np.sum(n * s * s) - np.sum(n * s) ** 2 / N)
    if V <= 0:
        return float("nan"), float("nan")
    z = T / math.sqrt(V)
    return z, math.erfc(abs(z) / math.sqrt(2.0))


def matthews(x, y):
    x = np.asarray(x, bool); y = np.asarray(y, bool)
    tp = np.sum(x & y); tn = np.sum(~x & ~y); fp = np.sum(~x & y); fn = np.sum(x & ~y)
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float("nan") if den == 0 else float(tp * tn - fp * fn) / den


# ------------------------------------------------------------- persistence
def persistence_flags(over, npers):
    """over: (n_eval, n_el) bool, r_e > rho per evaluation.
    Returns flagged[t, e]: True iff the last npers evaluations all exceeded."""
    over = np.asarray(over, bool)
    run = np.zeros(over.shape[1], dtype=int)
    out = np.zeros_like(over)
    for t in range(over.shape[0]):
        run = np.where(over[t], run + 1, 0)
        out[t] = run >= npers
    return out


# ----------------------------------------------------------- selection time
def time_selection(el, cfg, N, rho, rng, n_prof=400, warm=5):
    """Wall-clock time [ms] of one admissible-set + selection call, over
    n_prof random risk profiles.  The first `warm` calls (import, cache,
    allocator warm-up) are executed but discarded."""
    from masrecon.select import admissible_elements, select_structure
    out = []
    for r in range(n_prof + warm):
        rb = rng.uniform(0.0, 0.3, len(el))
        rb[rng.integers(len(el))] = rng.uniform(0.95, 1.0)
        for j in range(N):
            rb[el.index[("u", j, j)]] = 0.0          # actuators: tighten
        t0 = time.perf_counter_ns()
        A = admissible_elements(el, rb, rho, dead=[])
        select_structure(el, A, rb, list(el.all), cfg, N, dead=[],
                         return_stage=True)
        dt = time.perf_counter_ns() - t0
        if r >= warm:
            out.append(dt * 1e-6)
    return np.array(out)


# -------------------------------------------------------------- bookkeeping
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_hash(path):
    hx = sha256_file(path)
    with open(path + ".sha256", "w", encoding="utf-8") as f:
        f.write("%s  %s\n" % (hx, path))
    return hx


def env_info():
    import matplotlib
    return ("Python %s | NumPy %s | matplotlib %s | %s | %s"
            % (platform.python_version(), np.__version__, matplotlib.__version__,
               platform.platform(), platform.processor() or "cpu?"))
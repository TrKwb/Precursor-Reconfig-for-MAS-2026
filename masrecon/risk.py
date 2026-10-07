# -*- coding: utf-8 -*-
"""Task A : precursor-risk quantification (vectorised, ~1 ms per window).

Features : permutation entropy, RQA (RR/DET/LAM), log-std, lag-1 autocorrelation
Score    : Mahalanobis distance from the nominal feature distribution
Limit    : split-conformal upper confidence limit  rbar_e(alpha_e) = 1 - p_e

Validity : for an exchangeable healthy calibration set,
           P[p_e <= alpha_e] <= alpha_e   (marginal, per element per test).
           A union-bound allocation alpha_e = alpha/|C| gives family-wise alpha.
Caveat   : overlapping windows within one run are NOT exchangeable.  We
           calibrate on non-overlapping windows from independent healthy runs
           and report the empirical false-alarm rate as a direct check.
"""
import math
import numpy as np
from itertools import permutations

_PID = {}


def perm_entropy(x, order=4, delay=1):
    """Normalised permutation entropy (Bandt and Pompe, 2002)."""
    n = len(x) - (order - 1) * delay
    if n <= 1:
        return 0.0
    if order not in _PID:
        tab = np.zeros((order,) * order, dtype=np.int32)
        for i, p in enumerate(permutations(range(order))):
            tab[p] = i
        _PID[order] = tab
    tab = _PID[order]
    idx = np.arange(order) * delay
    emb = x[np.arange(n)[:, None] + idx[None, :]]
    ranks = np.argsort(np.argsort(emb, axis=1, kind="stable"), axis=1)
    cnt = np.bincount(tab[tuple(ranks.T)], minlength=tab.size)
    p = cnt[cnt > 0] / n
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))


def rqa2(x, m=3, tau=2, rr_target=0.05):
    """RR, DET, LAM with minimum line length two, exact and fully vectorised.

    A recurrence point lies on a diagonal (vertical) line of length at least two
    if and only if it has a neighbour in that direction, so both quantities
    follow from two shifted Boolean AND operations on the recurrence matrix.
    """
    n = len(x) - (m - 1) * tau
    if n < 8:
        return 0.0, 0.0, 0.0
    idx = np.arange(m) * tau
    Z = x[np.arange(n)[:, None] + idx[None, :]]
    sq = (Z * Z).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    R = D2 <= np.quantile(D2, rr_target)
    np.fill_diagonal(R, False)
    tot = R.sum()
    if tot == 0:
        return 0.0, 0.0, 0.0
    dn = np.zeros_like(R); dn[1:, 1:] = R[:-1, :-1]
    up = np.zeros_like(R); up[:-1, :-1] = R[1:, 1:]
    vd = np.zeros_like(R); vd[1:, :] = R[:-1, :]
    vu = np.zeros_like(R); vu[:-1, :] = R[1:, :]
    return (float(R.mean()),
            float((R & (dn | up)).sum() / tot),
            float((R & (vd | vu)).sum() / tot))


def features(w, rc):
    """Five-dimensional feature vector of one window."""
    w = np.asarray(w, float)
    sd = w.std()
    wn = w / sd if sd > 1e-12 else w
    pe = perm_entropy(wn, rc.pe_order, rc.pe_delay)
    rr, det, lam = rqa2(wn, rc.emb_m, rc.emb_tau, rc.rr_target)
    if sd > 1e-12:
        a = w[:-1] - w[:-1].mean()
        b = w[1:] - w[1:].mean()
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        ac1 = float((a * b).sum() / den) if den > 1e-12 else 0.0
    else:
        ac1 = 0.0
    return np.array([pe, det, lam, math.log(sd + 1e-9), ac1])


def features_many(Wm, rc):
    return np.stack([features(w, rc) for w in Wm])


class ConformalRisk:
    """Split-conformal anomaly detector.

        p_e    = (1 + #{s_cal >= s}) / (n_cal + 1)
        rbar_e = 1 - p_e            in  [0, 1 - 1/(n_cal+1)]

    Note that alpha_e >= 1/(n_cal+1) is necessary for the detector to be able
    to fire at all; build_calibration() enforces the corresponding lower bound
    on n_cal.
    """

    def __init__(self, rc):
        self.rc = rc

    def fit(self, cal_windows):
        F = features_many(cal_windows, self.rc)
        self.mu = F.mean(0)
        C = np.cov(F.T) + 1e-6 * np.eye(F.shape[1])
        self.Pinv = np.linalg.inv(C)
        d = F - self.mu
        self.cal = np.sort(np.sqrt(np.maximum((d @ self.Pinv * d).sum(1), 0.0)))
        self.n_cal = len(self.cal)
        return self

    def score_feats(self, F):
        d = F - self.mu
        return np.sqrt(np.maximum((d @ self.Pinv * d).sum(1), 0.0))

    def pvalues(self, s):
        s = np.atleast_1d(s)
        ge = self.n_cal - np.searchsorted(self.cal, s, side="left")
        return (1.0 + ge) / (self.n_cal + 1.0)

    def rbar_from_feats(self, F):
        return 1.0 - self.pvalues(self.score_feats(F))

    def rbar(self, w):
        return float(self.rbar_from_feats(features(w, self.rc)[None, :])[0])

    @property
    def rbar_max(self):
        return 1.0 - 1.0 / (self.n_cal + 1.0)


def risk_trace(plant, cr, rc):
    """rbar_e(k) on the stride grid.  The diagnostic signals are exogenous, so
    one trace is shared by all control methods and the comparison is paired."""
    K, M = plant.K + 1, plant.y.shape[1]
    ks = np.arange(rc.W, K, rc.stride)
    R = np.zeros((len(ks), M))
    for a, k in enumerate(ks):
        R[a] = cr.rbar_from_feats(
            features_many([plant.y[k - rc.W:k, j] for j in range(M)], rc))
    return ks, R


def build_calibration(cfg, Elements_cls, N, runs=30):
    """Calibrate on non-overlapping windows from independent healthy runs, so
    that the exchangeability hypothesis of Proposition 1 is defensible."""
    from .plant import Plant
    el = Elements_cls(N)
    rc = cfg.risk
    cal = []
    for t in range(runs):
        p = Plant(cfg, el, failing=[], rng=np.random.default_rng(50000 + t))
        for j in range(len(el)):
            for k0 in range(0, p.K - rc.W, rc.W):
                cal.append(p.y[k0:k0 + rc.W, j])
    if len(cal) < rc.n_cal:
        raise ValueError("need >= %d calibration windows, got %d "
                         "(increase runs= or reduce n_cal)"
                         % (rc.n_cal, len(cal)))
    return ConformalRisk(rc).fit(cal[:rc.n_cal])
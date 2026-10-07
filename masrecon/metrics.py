# -*- coding: utf-8 -*-
import numpy as np

def recovery_time(H, k_fail, dt, thresh=0.15, hold=1.0):
    t, err = H["t"], H["err"]
    i0 = int(np.searchsorted(t, k_fail * dt)); nh = int(round(hold / dt))
    for i in range(i0, len(t) - nh):
        if np.all(err[i:i + nh] < thresh):
            return float(t[i] - k_fail * dt)
    return np.nan

def safety_stats(H):
    h = H["h"]
    return dict(min_h=float(h.min()),
                nviol=int((h.min(axis=1) < 0).sum()),
                viol_area=float(np.clip(-h, 0, None).sum()),
                nslack=int((H["slack"] > 1e-6).sum()))

def detection_stats(ks, R, jf, rho, persist, k_fail, dt, M):
    def first(v):
        run = 0
        for a, z in enumerate(v):
            run = run + 1 if z > rho else 0
            if run >= persist:
                return a
        return -1
    a = first(R[:, jf])
    det = (a >= 0 and ks[a] < k_fail)
    oth = [j for j in range(M) if j != jf]
    return dict(detected=bool(det),
                lead=float((k_fail - ks[a]) * dt) if det else np.nan,
                far_check=float((R[:, oth] > rho).mean()),
                far_trial=bool(any(first(R[:, j]) >= 0 for j in oth)))

def summarize(rows):
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows
             if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not v:
            out[k] = (float("nan"), 0.0); continue
        if isinstance(v[0], (bool, np.bool_)):
            out[k] = (float(np.mean(v)), 0.0)
        else:
            out[k] = (float(np.mean(v)),
                      float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    return out
# -*- coding: utf-8 -*-
"""Diagnose runs dumped by runner.run (MAS_DUMP).

report  : python tools/diag_dump.py --dir results/dump_c1 --legacy C5
timeline: python tools/diag_dump.py --dir results/dump_c1 --legacy C5 --trial 3fa9 \
              --labels T_stop,T_rel10,C5 --from -40 --to 200 --every 4 --plot t.png
"""
import argparse, collections, glob, os, pickle
import numpy as np

TUBE = {"comm": "c", "drift": "d", "stop": "s", "stopdead": "D", "env": "e",
        "reach": "r", "dead": "x", "": "-"}


def label(meta, legacy):
    f = meta["flags"]
    if f.get("policy_T"):
        lr = f.get("latch_release")
        if lr is not None:
            return "T_rel%d" % int(lr)
        return "T_stop" if f.get("hard_barrier") else "T_soft"
    if meta["method"] != "prop":
        return meta["method"]
    return legacy or ("L_" + meta["cond"])


def load(d, legacy):
    runs = collections.defaultdict(dict)
    files = sorted(glob.glob(os.path.join(d, "*.pkl")))
    if not files:
        raise SystemExit("no pkl in %s" % d)
    for fn in files:
        with open(fn, "rb") as fh:
            D = pickle.load(fh)
        m = D["meta"]
        if "flag_all" not in D["H"]:
            raise SystemExit("%s has no diagnostic logs (patch runner.py first)" % fn)
        lab = label(m, legacy)
        if lab in runs[m["trial"]]:
            raise SystemExit("trial %s: two conditions map to %r; dump one legacy "
                             "condition only, or omit --legacy" % (m["trial"], lab))
        runs[m["trial"]][lab] = D
    return runs


def ferr_series(H):                      # err = ferr + 2*viol  ->  ferr
    return H["err"] - 2.0 * np.maximum(0.0, -H["h"].min(axis=1))


def uidx(H):
    names = [tuple(e) for e in H["el_names"]]
    return np.array([names.index(("u", j, j)) for j in range(H["p"].shape[1])])


def fail_of(D):
    fi = D["meta"]["fail"]
    return [tuple(e) for e in fi.get("failing", [])], fi.get("k_fail")


def streaks(H, j):
    """Unflagged-evaluation streak of actuator j at each step (-1: never flagged yet).
    Flags come from the pre-computed risk trace -> identical in every condition.
    Under latch_release=n the actuator is latched iff 0 <= streak < n."""
    ne, fj = H["new_eval"], H["flag_all"][:, uidx(H)[j]]
    s, out = -1, np.empty(len(ne), dtype=int)
    for k in range(len(ne)):
        if ne[k]:
            if fj[k]:
                s = 0
            elif s >= 0:
                s += 1
        out[k] = s
    return out


def summarize(D, thr, tail):
    H = D["H"]
    fails, kf = fail_of(D)
    N = H["p"].shape[1]
    dt = float(H["t"][1] - H["t"][0])
    fe = ferr_series(H)
    hmin = H["h"].min(axis=1)
    post = fe[kf:]
    bad = np.flatnonzero(post >= thr)
    rec = 0.0 if len(bad) == 0 else (None if bad[-1] == len(post) - 1
                                     else (bad[-1] + 1) * dt)
    idx = np.argsort(fe[:kf])[:max(5, kf // 10)]          # calmest pre-failure steps
    c = float(np.median(H["h"][idx]))                     # ~ spacing - d_min
    hend = H["h"][-tail:].mean(axis=0)
    g = int(np.argmax(np.abs(hend - c)))
    L = H["latched"]
    fu = H["flag_all"][:, uidx(H)]
    return dict(
        kind=fails[0][0] if fails else "-", elem=fails[0] if fails else None,
        kf=kf, dt=dt, viol=bool((hmin < 0).any()),
        nr=rec is None, rec=rec, ferr_end=float(fe[-tail:].mean()),
        v_end=float(H["v"][-tail:].mean()),
        latched_end=[j for j in range(N) if L[-1, j]],
        first_latch={j: (int(np.argmax(L[:, j])) - kf) * dt
                     for j in range(N) if L[:, j].any()},
        postfail_uflag={j: int(fu[kf:, j].sum()) for j in range(N) if fu[kf:, j].any()},
        worst_gap=(g, g + 1), worst_err=float(abs(hend[g] - c)),
        fwd_latched=bool(L[-1, g + 1]),
        tube_end="".join(TUBE.get(t, "?") for t in H["tube_fwd"][-1]),
        shield_post=int(H["shield"][kf:].sum()),
        qpfail_k=[int(k) - kf for k in np.flatnonzero((~H["qp_ok"]).any(axis=1))],
        nrr_k=[int(k) - kf for k in np.flatnonzero(np.diff(H["nrr_cum"], prepend=0) > 0)],
        hr_post=int(H["hr_cum"][-1] - H["hr_cum"][kf - 1]),
    )


def report(runs, a):
    S = {tr: {lab: summarize(D, a.thr, a.tail) for lab, D in labs.items()}
         for tr, labs in runs.items()}
    labs_all = sorted({l for v in S.values() for l in v})

    print("== 1. nr / viol by fault kind (thr=%.3f; match exp9's nr to calibrate) ==" % a.thr)
    for lab in labs_all:
        cnt, nr, vi = collections.Counter(), collections.Counter(), collections.Counter()
        for v in S.values():
            if lab in v:
                s = v[lab]; cnt[s["kind"]] += 1; nr[s["kind"]] += s["nr"]; vi[s["kind"]] += s["viol"]
        print("  %-8s " % lab + "   ".join("%s: nr %2d viol %2d / %2d" % (k, nr[k], vi[k], cnt[k])
                                         for k in sorted(cnt)))

    ref, cmp_ = a.ref, a.cmp
    print("\n== 2. %s nr trials  (H: latch never released + group still moving) ==" % ref)
    print("  %-12s %-12s %-5s %-10s %-24s %-14s %-7s %-4s %-6s %-6s %s"
          % ("trial", "fault", cmp_[:5], "latch_end", "first_latch[s rel kf]",
             "postfail_uflag", "worst", "fwdL", "v_end", "ferr", "tube_end"))
    agg = collections.Counter()
    for tr, v in sorted(S.items()):
        if ref not in v or not v[ref]["nr"]:
            continue
        s, o = v[ref], v.get(cmp_)
        fl = {j: round(t, 1) for j, t in s["first_latch"].items()}
        print("  %-12s %-12s %-5s %-10s %-24s %-14s %-7s %-4s %6.2f %6.3f %s"
              % (tr, s["elem"], "-" if o is None else ("nr" if o["nr"] else "ok"),
                 s["latched_end"], fl, s["postfail_uflag"], s["worst_gap"],
                 int(s["fwd_latched"]), s["v_end"], s["ferr_end"], s["tube_end"]))
        agg["n"] += 1
        agg["any_latch_end"] += bool(s["latched_end"])
        agg["worst_behind_latched"] += s["fwd_latched"]
        agg["end_latches_all_prefail"] += bool(s["latched_end"]) and all(
            s["first_latch"][j] <= 0 for j in s["latched_end"])
        agg["moving"] += s["v_end"] > 0.5
        agg["cmp_nr"] += bool(o and o["nr"])
    print("  summary:", dict(agg))
    ctr = collections.Counter()                       # sufficiency check
    for v in S.values():
        if ref in v:
            s = v[ref]
            if s["latched_end"] and s["v_end"] > 0.5:
                ctr[(s["kind"], "nr" if s["nr"] else "recovered")] += 1
    print("  %s runs with latch at end and group moving:" % ref, dict(ctr))

    print("\n== 3. E under latch_release=n, offline from the flag trace (u faults, L*=%d) ==" % a.Lstar)
    rows, chk = [], collections.Counter()
    for tr, labs in runs.items():
        D = next(iter(labs.values()))
        fails, kf = fail_of(D)
        if not fails or fails[0][0] != "u":
            continue
        j = fails[0][1]
        st = streaks(D["H"], j)
        w = st[max(0, kf - a.Lstar): kf + 1]
        wmax = None if (w < 0).any() else int(w.max())
        rows.append((tr, j, wmax))
        for lab, DD in labs.items():                  # cross-check vs logged latch
            if lab.startswith("T_"):
                n = None if lab in ("T_stop", "T_soft") else int(lab[5:])
                pred = wmax is not None and (n is None or wmax < n)
                obs = bool(DD["H"]["latched"][max(0, kf - a.Lstar): kf + 1, j].all())
                chk[(lab, pred == obs)] += 1
    print("  cross-check predicted vs logged E:", dict(chk), "(all must be True)")
    for sub, name in ((rows, "all u"), ([r for r in rows if r[1] >= 1], "jf>=1")):
        line = "  %-6s n=%3d  stop:%3d" % (name, len(sub), sum(r[2] is not None for r in sub))
        for n in (1, 2, 3, 5, 8, 10, 15, 20, 30):
            line += "  rel%d:%d" % (n, sum(r[2] is not None and r[2] < n for r in sub))
        print(line)
    ws = [r[2] for r in rows if r[2] is not None]
    if ws:
        print("  max unflagged streak inside E-window over u runs: %d  (any n > this keeps E = stop)" % max(ws))

    print("\n== 4. violation discordance ==")
    for x in labs_all:
        for y in labs_all:
            if x < y:
                xo = [tr for tr, v in S.items() if x in v and y in v and v[x]["viol"] and not v[y]["viol"]]
                yo = [tr for tr, v in S.items() if x in v and y in v and v[y]["viol"] and not v[x]["viol"]]
                if xo or yo:
                    print("  %s only: %s | %s only: %s" % (x, xo, y, yo))

    print("\n== 5. qpfail / nonretreat_relaxed (k relative to k_fail) ==")
    for tr, v in sorted(S.items()):
        for lab, s in sorted(v.items()):
            if s["qpfail_k"] or s["nrr_k"]:
                print("  %-12s %-8s %-12s qpfail@%s nrr@%s nr=%s latch_end=%s"
                      % (tr, lab, s["elem"], s["qpfail_k"], s["nrr_k"], s["nr"], s["latched_end"]))


def timeline(runs, a):
    c = [t for t in runs if t.startswith(a.trial)]
    if len(c) != 1:
        raise SystemExit("trial prefix %r matches %d trials" % (a.trial, len(c)))
    tr = c[0]
    labs = a.labels.split(",") if a.labels else sorted(runs[tr])
    m = lambda b: "".join("1" if x else "." for x in b)
    for lab in labs:
        D = runs[tr][lab]; H = D["H"]; fails, kf = fail_of(D); ui = uidx(H)
        fe = ferr_series(H)
        k0, k1 = max(0, kf + a.k_from), min(len(H["t"]), kf + a.k_to)
        print("\n--- %s  %s  fail %s  k_fail %d (t=%.2f) ---" % (tr, lab, fails, kf, H["t"][kf]))
        print("   t     ferr   hmin  @g  latched  flag_u   quiet         shield qpF  tube  crit oth  v | u")
        for k in range(k0, k1, a.every):
            oth = int(H["flag_all"][k].sum() - H["flag_all"][k, ui].sum())
            print("%s%6.2f %6.3f %7.3f %2d  %s %s %-13s %s %s %s %3d %3d  %s | %s" % (
                "*" if k <= kf < k + a.every else ("e" if H["new_eval"][k] else " "),
                H["t"][k], fe[k], H["h"][k].min(), int(np.argmin(H["h"][k])),
                m(H["latched"][k]), m(H["flag_all"][k, ui]),
                ",".join(str(int(q)) for q in H["quiet"][k]),
                m(H["shield_mask"][k]), m(~H["qp_ok"][k]),
                "".join(TUBE.get(t, "?") for t in H["tube_fwd"][k]),
                int(H["crit_in"][k]), oth,
                " ".join("%5.2f" % x for x in H["v"][k]),
                " ".join("%5.2f" % x for x in H["u"][k])))
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(11, 10))
        for lab in labs:
            H = runs[tr][lab]["H"]
            ax[0].plot(H["t"], ferr_series(H), label=lab)
            ax[1].plot(H["t"], H["h"].min(axis=1), label=lab)
            ax[2].plot(H["t"], H["v"].mean(axis=1), label=lab)
        Tl = next((l for l in labs if l.startswith("T_")), labs[0])
        H = runs[tr][Tl]["H"]; N = H["p"].shape[1]
        img = np.vstack([H["flag_all"][:, uidx(H)].T, H["latched"].T]).astype(float)
        ax[3].imshow(img, aspect="auto", interpolation="nearest", cmap="Greys",
                     extent=[H["t"][0], H["t"][-1], 2 * N - 0.5, -0.5])
        ax[3].set_yticks(range(2 * N))
        ax[3].set_yticklabels(["F%d" % j for j in range(N)] + ["L%d(%s)" % (j, Tl) for j in range(N)])
        kf = fail_of(runs[tr][labs[0]])[1]
        for x, yl in zip(ax, ("ferr", "h_min", "mean v", "")):
            x.axvline(H["t"][kf], color="r", ls="--"); x.set_ylabel(yl)
        ax[1].axhline(0, color="k", lw=0.5); ax[0].legend()
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print("saved", a.plot)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--legacy", default=None, help="name for the single legacy prop condition (e.g. C5)")
    p.add_argument("--thr", type=float, default=0.1, help="recovery threshold on ferr")
    p.add_argument("--tail", type=int, default=20)
    p.add_argument("--Lstar", type=int, default=2)
    p.add_argument("--ref", default="T_stop")
    p.add_argument("--cmp", default="T_rel10")
    p.add_argument("--trial", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--from", dest="k_from", type=int, default=-40)
    p.add_argument("--to", dest="k_to", type=int, default=200)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--plot", default=None)
    a = p.parse_args()
    runs = load(a.dir, a.legacy)
    timeline(runs, a) if a.trial else report(runs, a)
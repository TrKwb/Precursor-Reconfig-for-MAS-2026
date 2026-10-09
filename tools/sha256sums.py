# -*- coding: utf-8 -*-
"""Write / verify results/SHA256SUMS.txt (full 64-hex SHA-256).

    python tools/sha256sums.py write     # (re)generate the list
    python tools/sha256sums.py verify    # check files against the list
"""
import hashlib, os, sys

FILES = [
    "results/exp1.npz",
    "results/exp1_ws0.npz",
    "results/exp1_ws2.npz",
    "results/exp4_campaign2.npz",
    "results/exp8_wsafety.npy",
    "results/exp8_wsafety.csv",
    "results/exp2b_seltime.npz",
    "results/exp3b_conformal.npz",
    "results/exp5_mechanism.npy",
    "results/smoke_risk.npz",
    "results/smoke_fixed.npz",
    "results/smoke_post.npz",
    "results/smoke_prop.npz",
]
LIST = "results/SHA256SUMS.txt"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write():
    with open(LIST, "w", encoding="utf-8", newline="\n") as f:
        for p in FILES:
            if os.path.exists(p):
                f.write("%s  %s\n" % (sha256(p), p))
            else:
                print("missing (skipped):", p)
    print("wrote", LIST)


def verify():
    bad = 0
    for line in open(LIST, encoding="utf-8"):
        h, p = line.split(None, 1)
        p = p.strip()
        ok = os.path.exists(p) and sha256(p) == h
        print("%-4s %s" % ("OK" if ok else "FAIL", p))
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    {"write": write, "verify": verify}[sys.argv[1]]()
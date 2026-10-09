# -*- coding: utf-8 -*-
"""usage: python tools/grep.py FILE PATTERN [CONTEXT]"""
import sys
path, pat = sys.argv[1], sys.argv[2]
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 3
lines = open(path, encoding="utf-8").read().splitlines()
for i, s in enumerate(lines):
    if pat in s:
        lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
        for j in range(lo, hi):
            print("%5d%s %s" % (j + 1, ">" if j == i else " ", lines[j]))
        print("-" * 60)
# -*- coding: utf-8 -*-
"""usage: python tools/cmp_conditions.py FILE.npz CONDA CONDB"""
import sys
import numpy as np
d = np.load(sys.argv[1]); a, b = sys.argv[2], sys.argv[3]
ka = sorted(k for k in d.files if k.startswith(a + "_"))
for k in ka:
    kb = b + k[len(a):]
    if kb in d.files:
        x, y = d[k], d[kb]
        same = x.shape == y.shape and np.allclose(x, y, equal_nan=True)
        print("%-40s %s" % (k[len(a) + 1:], "same" if same else "DIFF"))
print("keys:", d.files[:8], "...")
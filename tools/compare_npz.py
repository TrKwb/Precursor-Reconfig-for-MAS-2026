# -*- coding: utf-8 -*-
"""Compare two .npz files array by array, ignoring timing fields."""
import sys
import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
skip = ("sel_ms",)                       # wall-clock fields
ka, kb = set(a.files), set(b.files)
print("only in A:", sorted(ka - kb))
print("only in B:", sorted(kb - ka))
bad = 0
for k in sorted(ka & kb):
    if any(s in k for s in skip):
        continue
    x, y = a[k], b[k]
    same = x.shape == y.shape and np.allclose(x, y, rtol=0, atol=1e-9,
                                              equal_nan=True)
    if not same:
        bad += 1
        print("DIFF", k, x.shape, y.shape)
print("RESULT:", "IDENTICAL (excluding timing)" if bad == 0 else "%d arrays differ" % bad)

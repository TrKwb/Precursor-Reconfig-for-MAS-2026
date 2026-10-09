"""tools/check_hr.py : dump の全 pkl で hr_cum の単調性と hr_cum[-1]==hard_relaxed を確認"""
import glob, pickle, collections
import numpy as np

def find(o, key):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            else: yield from find(v, key)
    elif isinstance(o, (list, tuple)):
        for v in o: yield from find(v, key)

files = sorted(glob.glob("results/dump_c1/*.pkl"))
cnt = collections.Counter()
for f in files:
    d = pickle.load(open(f, "rb"))
    hc, hr = list(find(d, "hr_cum")), list(find(d, "hard_relaxed"))
    if not hc:
        cnt["hr_cum なし"] += 1; continue
    for i, h in enumerate(hc):
        h = np.asarray(h); dh = np.diff(h, prepend=0)
        cnt["系列"] += 1
        if np.any(dh < 0):  cnt["減少あり"] += 1; print("減少:", f)
        if np.any(dh > 1):  cnt["1 step で 2 回以上 relax"] += 1
        if i < len(hr) and int(h[-1]) != int(hr[i]):
            cnt["末尾不一致"] += 1; print("不一致:", f, h[-1], hr[i])
print(len(files), "files", dict(cnt))
"""tools/check_subset.py
v11 onset から dbg と同じ shift の行を取り出し、共通キーをビット単位で比較する。
"""
import sys
import numpy as np

pa = "results/exp9_onset_v11.npz"
pb = "results/exp9_onset_dbg.npz"
A = np.load(pa, allow_pickle=True)
B = np.load(pb, allow_pickle=True)

print("v11 shift:", dict(zip(*np.unique(A["shift"], return_counts=True))))
print("dbg shift:", dict(zip(*np.unique(B["shift"], return_counts=True))))
sb = np.unique(B["shift"])
if len(sb) != 1:
    print("[NG] dbg に複数の shift がある"); sys.exit(0)

m = np.flatnonzero(A["shift"] == sb[0])
print(f"取り出した行数: {len(m)} / dbg {len(B['shift'])}")

# run の対応づけ（labels と si の組で照合）
la = list(zip(A["labels"][m].tolist(), A["si"][m].tolist()))
lb = list(zip(B["labels"].tolist(), B["si"].tolist()))
if la != lb:
    if sorted(la) != sorted(lb) or len(set(la)) != len(la):
        print("[NG] run の集合が一致しない（または labels,si が一意でない）")
        print("  v11 のみ:", sorted(set(la) - set(lb))[:10])
        print("  dbg のみ:", sorted(set(lb) - set(la))[:10])
        sys.exit(0)
    pos = {key: i for i, key in enumerate(la)}
    m = m[[pos[key] for key in lb]]
    print("並び順が違うので labels,si で並べ替えた")

def bits(x):
    x = np.ascontiguousarray(x)
    return x.view(f"u{x.dtype.itemsize}") if x.dtype.kind == "f" else x

common = sorted(set(A.files) & set(B.files))
ng = 0
for k in common:
    a, b = A[k][m], B[k]
    if a.dtype != b.dtype or a.shape != b.shape:
        print(f"[NG] {k}: {a.shape},{a.dtype} vs {b.shape},{b.dtype}"); ng += 1; continue
    d = np.flatnonzero(bits(a) != bits(b))
    if len(d):
        ng += 1
        print(f"[NG] {k}: 不一致 {len(d)}/{len(b)}")
        for i in d[:5]:
            print(f"      run {B['labels'][i]}: {a[i]!r} vs {b[i]!r}")
print("=" * 60)
print(f"共通キー {len(common)} 個")
print("VERDICT:", "ALL IDENTICAL" if ng == 0 else f"{ng} KEY(S) DIFFER")
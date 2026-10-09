"""tools/check_identical.py
2 つの npz を、共通キーすべてについてビット単位で比較する。
usage: python tools/check_identical.py A.npz B.npz
  - float は -0.0 / 0.0 を区別し、同じビットの NaN は一致とみなす
  - object 配列（dict / list / tuple の入れ子）は再帰的に比較する
  - 片方にしかないキー（_dbg で追加したログなど）は一覧表示のみ
  - py() ヘルパー（check=True）で落ちないよう、終了コードは常に 0
"""
import sys, struct
import numpy as np

MAX_SHOW = 5

def _num_diff_idx(a, b):
    a = np.ascontiguousarray(a).ravel()
    b = np.ascontiguousarray(b).ravel()
    if a.dtype.kind == "c":                       # complex -> 実部/虚部の float として扱う
        a, b = a.view(a.real.dtype), b.view(b.real.dtype)
    if a.dtype.kind == "f":                       # float -> 同じ幅の uint で比較
        u = f"u{a.dtype.itemsize}"
        return np.flatnonzero(a.view(u) != b.view(u)), a, b
    return np.flatnonzero(a != b), a, b

def _fmt(x):
    try:
        return float(x).hex()
    except Exception:
        return repr(x)[:80]

def same(a, b, path, msgs):
    # numpy 配列
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            msgs.append(f"{path}: shape/dtype {a.shape},{a.dtype} vs {b.shape},{b.dtype}")
            return False
        if a.dtype == object:
            ok = True
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                if not same(x, y, f"{path}[{i}]", msgs):
                    ok = False
            return ok
        idx, fa, fb = _num_diff_idx(a, b)
        for i in idx[:MAX_SHOW]:
            msgs.append(f"{path} flat[{i}]: {_fmt(fa[i])} vs {_fmt(fb[i])}")
        if len(idx) > MAX_SHOW:
            msgs.append(f"{path}: ... 他 {len(idx) - MAX_SHOW} 箇所")
        return len(idx) == 0
    # dict
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            msgs.append(f"{path}: dict keys differ  A-B={sorted(map(str, set(a)-set(b)))}  B-A={sorted(map(str, set(b)-set(a)))}")
            return False
        ok = True
        for k in a:
            if not same(a[k], b[k], f"{path}[{k!r}]", msgs):
                ok = False
        return ok
    # list / tuple
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            msgs.append(f"{path}: {type(a).__name__}[{len(a)}] vs {type(b).__name__}[{len(b)}]")
            return False
        ok = True
        for i, (x, y) in enumerate(zip(a, b)):
            if not same(x, y, f"{path}[{i}]", msgs):
                ok = False
        return ok
    # float スカラー
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if struct.pack("<d", float(a)) != struct.pack("<d", float(b)):
            msgs.append(f"{path}: {_fmt(a)} vs {_fmt(b)}")
            return False
        return True
    # その他（int, str, bool, None ...）
    try:
        eq = bool(a == b) and type(a) is type(b)
    except Exception:
        eq = repr(a) == repr(b)
    if not eq:
        msgs.append(f"{path}: {repr(a)[:80]} vs {repr(b)[:80]}")
    return eq

def main(pa, pb):
    A = np.load(pa, allow_pickle=True)
    B = np.load(pb, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    common = sorted(ka & kb)
    print(f"A = {pa}\nB = {pb}")
    print(f"共通キー {len(common)} / A のみ {len(ka - kb)} / B のみ {len(kb - ka)}")
    if ka - kb: print("  A のみ:", sorted(ka - kb))
    if kb - ka: print("  B のみ:", sorted(kb - ka))
    n_ng = 0
    for k in common:
        msgs = []
        ok = same(A[k], B[k], k, msgs)
        print(f"[{'OK' if ok else 'NG'}] {k}")
        for m in msgs[:3 * MAX_SHOW]:
            print("     ", m)
        n_ng += (not ok)
    print("=" * 60)
    print("VERDICT:", "ALL IDENTICAL" if n_ng == 0 else f"{n_ng} KEY(S) DIFFER")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
    else:
        main(sys.argv[1], sys.argv[2])
    sys.exit(0)
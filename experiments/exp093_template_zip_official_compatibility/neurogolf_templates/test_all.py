"""
test_all.py — 全テンプレート（基本+複合+論理+色置換）の
正しさ検証 ＋ onnx-tool 実測コスト測定（校正）
================================================================================
各テンプレートについて:
  1. ローカル実行で変換の正しさを確認（4x6グリッド）
  2. onnx-tool で実測 MAC / memory / params を測定
  3. cost = MAC + memory + params を算出し、250-600帯判定
"""
import io
import contextlib
import math
import numpy as np
import onnx

import templates as T
import templates_composite as C
import templates_logical as L
import templates_recolor_dynamic as R

# テスト用グリッド 4x6（非正方・偶数幅でXOR等もテスト可）
g = np.array([
    [0, 1, 2, 3, 0, 1],
    [0, 1, 0, 3, 4, 0],
    [5, 0, 0, 3, 4, 2],
    [5, 5, 0, 0, 4, 2],
], dtype=np.uint8)
H, W = g.shape


def measure(model, feed):
    """onnx-tool で (macs, memory, params) を実測。
       print_node_map の Total 行をパースして集計値を得る。"""
    import onnx_tool
    onnx.save(model, "/tmp/_m.onnx")
    try:
        m = onnx_tool.Model("/tmp/_m.onnx")
        m.graph.shape_infer(feed)
        m.graph.profile()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m.graph.print_node_map()
        # Total 行: "Total  _  <MAC>  100%  <mem>  100%  <params>  100% ..."
        macs = mem = params = 0
        for line in buf.getvalue().splitlines():
            if line.strip().startswith("Total"):
                nums = [t.replace(",", "") for t in line.split()
                        if t.replace(",", "").isdigit()]
                if len(nums) >= 3:
                    macs, mem, params = int(nums[0]), int(nums[1]), int(nums[2])
                break
        return macs, mem, params
    except Exception as e:
        return None, None, f"err:{e}"


def score(c):
    return max(1, 25 - math.log(c)) if c and c > 0 else 0


def check(name, model, expected, feed, run_fn):
    # 基本テンプレ(T.run)は ndarray を、複合/論理は dict を受ける。
    # run_fn が T.run/C.run/R.run（単一入力 'in'）なら ndarray を渡す。
    try:
        if run_fn in (T.run, C.run, R.run) and set(feed.keys()) == {"in"}:
            got = run_fn(model, feed["in"])
        else:
            got = run_fn(model, feed)
    except TypeError:
        got = run_fn(model, feed)
    ok = np.array_equal(got, expected)
    macs, mem, params = measure(model, feed)
    cost = (macs or 0) + (mem or 0) + (params if isinstance(params, int) else 0)
    band = "★250-600" if 250 <= cost <= 600 else (
        "(<250)" if cost < 250 else "(>600)")
    s = f"{score(cost):.2f}" if cost else "?"
    status = "OK " if ok else "NG!"
    print(f"[{status}] {name:22s} MAC={str(macs):>4} mem={str(mem):>4} "
          f"par={str(params):>3} cost={cost:>4} score={s} {band}")
    return ok


print("=" * 78)
print("全テンプレート 正しさ検証 + onnx-tool実測コスト (4x6グリッド)")
print("=" * 78)
res = []

print("\n--- 基本テンプレート (templates.py) ---")
res.append(check("identity", T.t_identity(), g, {"in": g}, T.run))
res.append(check("flip_lr", T.t_flip_lr(), g[:, ::-1], {"in": g}, T.run))
res.append(check("rot180", T.t_rot180(), g[::-1, ::-1], {"in": g}, T.run))
res.append(check("rot90_ccw", T.t_rot90_ccw(), np.rot90(g), {"in": g}, T.run))
res.append(check("transpose", T.t_transpose(), g.T, {"in": g}, T.run))
res.append(check("tile2x2", T.t_tile(2, 2), np.tile(g, (2, 2)), {"in": g}, T.run))
res.append(check("crop", T.t_crop(1, 3, 1, 4), g[1:3, 1:4], {"in": g}, T.run))
res.append(check("pad", T.t_pad(1, 1, 1, 1), np.pad(g, 1), {"in": g}, T.run))

print("\n--- 複合変換 (templates_composite.py) ---")
res.append(check("flip_both_fused", C.c_flip_both_fused(), g[::-1, ::-1],
                 {"in": g}, C.run))
res.append(check("mirror_tile_h", C.c_mirror_tile_h(),
                 np.concatenate([g, g[:, ::-1]], axis=1), {"in": g}, C.run))
res.append(check("crop_then_flip", C.c_crop_then_flip(1, 3, 1, 5),
                 g[1:3, 1:5][:, ::-1], {"in": g}, C.run))
# rot90 then recolor (int32入力)
mp = [0, 6, 4, 8, 4, 5, 6, 7, 8, 9]
rot = np.rot90(g)
exp = np.array([[mp[v] for v in row] for row in rot], dtype=np.uint8)
res.append(check("rot90_recolor", C.c_rot90_then_recolor(mp), exp,
                 {"in": g.astype(np.int32)}, C.run))
# chain: transpose -> flip axis0
ch = C.c_chain([("transpose",), ("flip", 0)])
res.append(check("chain_tr_flip", ch, np.rot90(g, -1)[::-1] if False else g.T[::-1],
                 {"in": g}, C.run))

print("\n--- 論理系 Tier B (templates_logical.py) ---")
# recolor_where: 色3を色7に
exp_rw = np.where(g == 3, 7, g).astype(np.uint8)
res.append(check("recolor_where", L.l_recolor_where(3, 7), exp_rw,
                 {"in": g}, L.run))
# halves_xor: 左右3列ずつ
Lh, Rh = g[:, :3], g[:, 3:]
exp_xor = np.where((Lh != 0) ^ (Rh != 0), 1, 0).astype(np.uint8)
res.append(check("halves_xor", L.l_halves_xor(1), exp_xor, {"in": g}, L.run))
# two_grids_and
a = (g > 0).astype(np.uint8) * 3
b = np.array([[1, 0, 1, 0, 1, 0]] * 4, np.uint8)
exp_and = np.where((a != 0) & (b != 0), 1, 0).astype(np.uint8)
res.append(check("two_grids_and", L.l_two_grids_and(1), exp_and,
                 {"a": a, "b": b}, lambda m, f: L.run(m, f)))

print("\n--- 色置換 動的形状 (templates_recolor_dynamic.py) ---")
exp_rc = np.array([[mp[v] for v in row] for row in g], dtype=np.uint8)
res.append(check("recolor_direct(int32)", R.recolor_direct(mp), exp_rc,
                 {"in": g.astype(np.int32)}, R.run))
res.append(check("recolor_cast(uint8)", R.recolor_cast(mp), exp_rc,
                 {"in": g}, R.run))

print()
print(f"=== 結果: {sum(res)}/{len(res)} テンプレート正常 ===")

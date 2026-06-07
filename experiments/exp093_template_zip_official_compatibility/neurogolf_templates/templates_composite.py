"""
templates_composite.py — 複合変換テンプレート（FREE op の合成）
================================================================================
複数の変換を連結する。中間テンソルが増える分 memory(=出力バイト) が積み増す。

【onnx-tool 実測校正済みの事実】
  - memory項 = 各ノードの「出力テンソル」バイトの総和（入力テンソルは計上されない）
  - よって複合変換のcost ≒ Σ(各ノード出力の要素数 × dtypeバイト)
  - ノード数を減らす = 中間出力を減らす = costを下げる（ノード融合の動機）

【ノード融合の指針】
  - 2つの軸反転(Slice)は1つのSlice(両軸同時)に融合できる → 中間1個削減
  - 連続するTranspose同士はpermを合成して1つに
  - flip→recolor のような異種opは融合不可だが、順序を選んで中間サイズを最小化
    （例: 先に crop で小さくしてから recolor すると中間が小さい）
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

U8 = TensorProto.UINT8
I32 = TensorProto.INT32
OPSET = 18


def _const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _slice_node(inp, out, axes, steps, name_prefix=""):
    """負ステップ対応の汎用Sliceノードと定数を返す。"""
    rank = len(axes)
    starts = [-1 if s < 0 else 0 for s in steps]
    ends = [-(1 << 31) if s < 0 else (1 << 31) for s in steps]
    p = name_prefix
    inits = [_const(f"{p}s", np.array(starts, np.int64)),
             _const(f"{p}e", np.array(ends, np.int64)),
             _const(f"{p}a", np.array(axes, np.int64)),
             _const(f"{p}st", np.array(steps, np.int64))]
    node = helper.make_node(
        "Slice", [inp, f"{p}s", f"{p}e", f"{p}a", f"{p}st"], [out])
    return node, inits


def _finalize(nodes, inits, name, in_dtype=U8, out_dtype=U8):
    inp = helper.make_tensor_value_info("in", in_dtype, ["H", "W"])
    out = helper.make_tensor_value_info("out", out_dtype, ["H2", "W2"])
    g = helper.make_graph(nodes, name, [inp], [out], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def run(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"in": x})[0]


# ---------------------------------------------------------------------------
# C1: 反転 + 反転 → 融合（180度回転と等価）
# ---------------------------------------------------------------------------
def c_flip_both_fused():
    """flip_lr + flip_ud を1つのSlice(両軸同時)に融合。中間テンソル0個。
       素朴に2ノードで書くと中間1個(+100B)。融合で削減。"""
    node, inits = _slice_node("in", "out", axes=[0, 1], steps=[-1, -1])
    return _finalize([node], inits, "flip_both_fused")


# ---------------------------------------------------------------------------
# C2: 90度回転 + 色置換（融合不可・順序最適化）
# ---------------------------------------------------------------------------
def c_rot90_then_recolor(mapping):
    """回転してから色置換。Transpose+Slice+Gather の3ノード。
       色置換のGatherはindicesが整数必須 → 入力int32で受ける。
       中間: t(転置), r(回転後) の2個 + 出力。"""
    t = helper.make_node("Transpose", ["in"], ["t"], perm=[1, 0])
    s_node, s_inits = _slice_node("t", "r", axes=[0], steps=[-1], name_prefix="r_")
    lut = _const("lut", np.asarray(mapping, np.uint8))
    g = helper.make_node("Gather", ["lut", "r"], ["out"], axis=0)
    return _finalize([t, s_node, g], s_inits + [lut],
                     "rot90_recolor", in_dtype=I32, out_dtype=U8)


# ---------------------------------------------------------------------------
# C3: crop して小さくしてから処理（中間最小化の順序最適化）
# ---------------------------------------------------------------------------
def c_crop_then_flip(r0, r1, c0, c1):
    """先にcropで領域を小さくしてからflip。
       逆順(flip→crop)より中間テンソルが小さくなる場合がある。
       ここでは crop(Slice) → flip(Slice) の2ノード。"""
    st = _const("cs", np.array([r0, c0], np.int64))
    en = _const("ce", np.array([r1, c1], np.int64))
    ax = _const("ca", np.array([0, 1], np.int64))
    crop = helper.make_node("Slice", ["in", "cs", "ce", "ca"], ["cropped"])
    flip, finits = _slice_node("cropped", "out", axes=[1], steps=[-1], name_prefix="f_")
    return _finalize([crop, flip], [st, en, ax] + finits, "crop_flip")


# ---------------------------------------------------------------------------
# C4: タイル + 反転の組み合わせ（鏡像タイル）
# ---------------------------------------------------------------------------
def c_mirror_tile_h():
    """[入力 | 左右反転] を横連結（対称展開）。Slice + Concat の2ノード。
       出力は横2倍。複合だが中間1個(flip結果)のみ。"""
    flip, finits = _slice_node("in", "m", axes=[1], steps=[-1])
    cat = helper.make_node("Concat", ["in", "m"], ["out"], axis=1)
    return _finalize([flip, cat], finits, "mirror_tile_h")


# ---------------------------------------------------------------------------
# C5: 汎用チェイン（任意のFREE opを順に適用するビルダー）
# ---------------------------------------------------------------------------
def c_chain(ops):
    """opsリストを順に連結する汎用ビルダー。
       ops: [("transpose",), ("flip",axis), ("crop",r0,r1,c0,c1), ("tile",rh,rw)]
       中間テンソル数 = len(ops)-1。融合余地があれば呼び出し側で削る。"""
    nodes, inits = [], []
    cur = "in"
    for i, op in enumerate(ops):
        nxt = "out" if i == len(ops) - 1 else f"t{i}"
        kind = op[0]
        if kind == "transpose":
            nodes.append(helper.make_node("Transpose", [cur], [nxt], perm=[1, 0]))
        elif kind == "flip":
            n, ini = _slice_node(cur, nxt, axes=[op[1]], steps=[-1], name_prefix=f"f{i}_")
            nodes.append(n); inits += ini
        elif kind == "crop":
            _, r0, r1, c0, c1 = op
            inits += [_const(f"cs{i}", np.array([r0, c0], np.int64)),
                      _const(f"ce{i}", np.array([r1, c1], np.int64)),
                      _const(f"ca{i}", np.array([0, 1], np.int64))]
            nodes.append(helper.make_node(
                "Slice", [cur, f"cs{i}", f"ce{i}", f"ca{i}"], [nxt]))
        elif kind == "tile":
            inits.append(_const(f"r{i}", np.array([op[1], op[2]], np.int64)))
            nodes.append(helper.make_node("Tile", [cur, f"r{i}"], [nxt]))
        else:
            raise ValueError(kind)
        cur = nxt
    return _finalize(nodes, inits, "chain")

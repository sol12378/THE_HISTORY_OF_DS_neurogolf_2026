"""
NeuroGolf Championship 2026 — uint8固定・最小テンソル構成 ONNXグラフ生成テンプレート集
================================================================================

設計方針（解釈A: 1タスクあたり cost 250-600 を狙う）:
  - cost = params(要素数) + MAC + memory(バイト)
  - FREE op のみ使用 → params≈0, MAC=0, cost ≒ memory(バイト総和)
  - 全テンソルを uint8 で通す（色値0-9はuint8で十分、int64比1/8）
  - 中間テンソル数を最小化（memory項 = 各テンソルのバイト総和）

各テンプレートの方針:
  - 入力 'in'  : uint8 [H, W]   (1チャンネル整数グリッド。one-hot展開しない)
  - 出力 'out' : uint8 [H', W']
  - データ移動 op (Transpose/Slice/Gather/Tile/Concat/Pad/Reshape) のみで構成
  - Castは使わず、最初からグラフ全体をuint8で定義（Cast自体もテンソルを増やすため）

注意: onnx-tool の正確なテンソル計上方法（入力を含むか、複数デモペアの扱い）は
      実測で校正すること。本テンプレートは「中間テンソル数を最小化する」原則に従う。
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

U8 = TensorProto.UINT8
OPSET = 18


# ---------------------------------------------------------------------------
# 共通ユーティリティ
# ---------------------------------------------------------------------------
def _const(name, arr):
    """initializer (定数テンソル) を作る。形状指定やindex用。"""
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _build(nodes, inits, name, out_shape=None):
    """グラフ→モデルを組み立て、checkして返す。入出力はuint8 [H,W]想定。"""
    inp = helper.make_tensor_value_info("in", U8, ["H", "W"])
    out = helper.make_tensor_value_info(
        "out", U8, out_shape if out_shape else ["H2", "W2"])
    graph = helper.make_graph(nodes, name, [inp], [out], initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def count_tensors(model):
    """グラフ内テンソル数（入力+初期化子+各ノード出力）を数える。
       memory項の概算根拠。実際の onnx-tool 計上とは要校正。"""
    g = model.graph
    names = set()
    for i in g.input:
        names.add(i.name)
    for init in g.initializer:
        names.add(init.name)
    for n in g.node:
        for o in n.output:
            names.add(o)          # node.output は文字列のリスト
    return len(names), len(g.node), len(g.initializer)


def run(model, x):
    """入力xでローカル実行。正しさ検証用。xのdtypeをそのまま使う。"""
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"in": x})[0]


# ===========================================================================
# テンプレート群（タスク類型ごと）
# ===========================================================================

# --- T1: 恒等 / コピー -----------------------------------------------------
def t_identity():
    """出力=入力。Identity 1ノード。"""
    n = helper.make_node("Identity", ["in"], ["out"])
    return _build([n], [], "identity")


# --- T2: 左右反転（水平フリップ） ------------------------------------------
def t_flip_lr():
    """W軸を負ステップSliceで反転。Reverse相当をparam0/MAC0で実現。"""
    starts = _const("s", [-1]); ends = _const("e", [-(1 << 31)])
    axes = _const("a", [1]);    steps = _const("st", [-1])
    n = helper.make_node("Slice", ["in", "s", "e", "a", "st"], ["out"])
    return _build([n], [starts, ends, axes, steps], "flip_lr")


# --- T3: 上下反転（垂直フリップ） ------------------------------------------
def t_flip_ud():
    starts = _const("s", [-1]); ends = _const("e", [-(1 << 31)])
    axes = _const("a", [0]);    steps = _const("st", [-1])
    n = helper.make_node("Slice", ["in", "s", "e", "a", "st"], ["out"])
    return _build([n], [starts, ends, axes, steps], "flip_ud")


# --- T4: 180度回転 ---------------------------------------------------------
def t_rot180():
    """両軸同時に負ステップSlice。1ノードで180度回転。"""
    starts = _const("s", [-1, -1]); ends = _const("e", [-(1 << 31), -(1 << 31)])
    axes = _const("a", [0, 1]);     steps = _const("st", [-1, -1])
    n = helper.make_node("Slice", ["in", "s", "e", "a", "st"], ["out"])
    return _build([n], [starts, ends, axes, steps], "rot180")


# --- T5: 90度回転（反時計） ------------------------------------------------
def t_rot90_ccw():
    """転置してから列(元の行)を反転。Transpose + Slice の2ノード。
       反時計90度: out[i,j] = in[j, W-1-i]  ≡ transpose後にaxis0を反転。"""
    t = helper.make_node("Transpose", ["in"], ["t"], perm=[1, 0])
    starts = _const("s", [-1]); ends = _const("e", [-(1 << 31)])
    axes = _const("a", [0]);    steps = _const("st", [-1])
    s = helper.make_node("Slice", ["t", "s", "e", "a", "st"], ["out"])
    return _build([t, s], [starts, ends, axes, steps], "rot90_ccw")


# --- T6: 90度回転（時計回り） ----------------------------------------------
def t_rot90_cw():
    """時計90度: 転置してから列(元の列)を反転 = axis1反転。"""
    t = helper.make_node("Transpose", ["in"], ["t"], perm=[1, 0])
    starts = _const("s", [-1]); ends = _const("e", [-(1 << 31)])
    axes = _const("a", [1]);    steps = _const("st", [-1])
    s = helper.make_node("Slice", ["t", "s", "e", "a", "st"], ["out"])
    return _build([t, s], [starts, ends, axes, steps], "rot90_cw")


# --- T7: 転置（主対角鏡映） ------------------------------------------------
def t_transpose():
    n = helper.make_node("Transpose", ["in"], ["out"], perm=[1, 0])
    return _build([n], [], "transpose")


# --- T8: 色置換（カラーマッピング） ----------------------------------------
def t_recolor(mapping):
    """色i -> mapping[i]。Gather(LUT, in)。LUTはparams=10(uint8)のみ、MAC0。
       mapping: 長さ10のリスト（0-9の置換先）。
       注意: Gatherのindicesは整数型必須。色置換タスクでは入力をint32で受ける
            （グリッド値そのものがLUTのindexになる）。出力はuint8。
       テンソル: in(int32) + lut(uint8) + out(uint8) = 最小構成。"""
    inp = helper.make_tensor_value_info("in", TensorProto.INT32, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    lut = _const("lut", np.asarray(mapping, dtype=np.uint8))
    n = helper.make_node("Gather", ["lut", "in"], ["out"], axis=0)
    graph = helper.make_graph([n], "recolor", [inp], [out], initializer=[lut])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


# --- T9: タイル複製（縦横にrep回） -----------------------------------------
def t_tile(rep_h, rep_w):
    """Tile で縦rep_h・横rep_w複製。出力が拡大する点に注意（memory増）。"""
    reps = _const("r", np.asarray([rep_h, rep_w], dtype=np.int64))
    n = helper.make_node("Tile", ["in", "r"], ["out"])
    return _build([n], [reps], "tile")


# --- T10: 切り出し（固定領域のSlice） --------------------------------------
def t_crop(r0, r1, c0, c1):
    """[r0:r1, c0:c1] を切り出す。1 Slice。"""
    starts = _const("s", np.asarray([r0, c0], dtype=np.int64))
    ends = _const("e", np.asarray([r1, c1], dtype=np.int64))
    axes = _const("a", np.asarray([0, 1], dtype=np.int64))
    n = helper.make_node("Slice", ["in", "s", "e", "a"], ["out"])
    return _build([n], [starts, ends, axes], "crop")


# --- T11: パディング（枠付け） ---------------------------------------------
def t_pad(top, bottom, left, right, value=0):
    """周囲をvalueでパディング。Pad 1ノード。"""
    pads = _const("p", np.asarray([top, left, bottom, right], dtype=np.int64))
    cval = _const("c", np.asarray(value, dtype=np.uint8))
    n = helper.make_node("Pad", ["in", "p", "c"], ["out"], mode="constant")
    return _build([n], [pads, cval], "pad")


# --- T12: 連結（入力を自身と結合） -----------------------------------------
def t_concat_self(axis):
    """入力を自分自身とaxis方向に連結（鏡像連結等の基礎）。Concat 1ノード。"""
    n = helper.make_node("Concat", ["in", "in"], ["out"], axis=axis)
    return _build([n], [], "concat_self")


# --- T13: 反転連結（入力 + その鏡像を連結 = 対称展開） ----------------------
def t_mirror_concat(axis):
    """入力 と その軸反転 を連結（対称パターン生成）。Slice + Concat の2ノード。"""
    starts = _const("s", [-1]); ends = _const("e", [-(1 << 31)])
    axes = _const("a", [axis]); steps = _const("st", [-1])
    flip = helper.make_node("Slice", ["in", "s", "e", "a", "st"], ["m"])
    cat = helper.make_node("Concat", ["in", "m"], ["out"], axis=axis)
    return _build([flip, cat], [starts, ends, axes, steps], "mirror_concat")


# --- T14: ダウンスケール（k間隔で間引き） ----------------------------------
def t_downscale(k):
    """ストライドkのSliceで縦横を1/kに間引く。1 Slice。"""
    starts = _const("s", np.asarray([0, 0], dtype=np.int64))
    ends = _const("e", np.asarray([1 << 31, 1 << 31], dtype=np.int64))
    axes = _const("a", np.asarray([0, 1], dtype=np.int64))
    steps = _const("st", np.asarray([k, k], dtype=np.int64))
    n = helper.make_node("Slice", ["in", "s", "e", "a", "st"], ["out"])
    return _build([n], [starts, ends, axes, steps], "downscale")


# --- T15: 整数アップスケール（各セルをk×k拡大） ----------------------------
def t_upscale(k):
    """各セルをk×kブロックに拡大。Reshape→Tile→Reshape→Transpose系ではなく、
       最小構成として2軸Tile後にブロック整列を行う方式。
       ここでは Resize を避け、データ移動のみで実装（MAC0）。
       手法: in[H,W] -> Unsqueeze 2軸 -> Tile -> Reshape で [H*k, W*k]。"""
    # in[H,W] -> [H,1,W,1]
    sh1 = _const("sh1", np.asarray([0, 1, 1, 1], dtype=np.int64))  # 使わない簡易版
    # 簡易・確実版: Tileで[H,W]->[H, W*k]を作るのではなくブロック展開が必要。
    # 最小ノードで厳密なk×k拡大: Reshape系は形状が動的だと難しいため、
    # ここでは「行方向Tile + 列方向Tile + 整列」をGatherで行う安全版を提供。
    # 行・列それぞれを各k回繰り返すindexをGatherで作る（indexはparam少量）。
    # 実装は task固有のHを要するため、ここではプレースホルダとして
    # 後段の make_upscale_for_shape を使う。
    raise NotImplementedError("use make_upscale_for_shape(H, W, k)")


def make_upscale_for_shape(H, W, k):
    """形状確定時のk×k整数アップスケール。Gather2回（行・列）でMAC0実装。
       行index = [0,0,..(k),1,1,.., H-1..]、列も同様。indexはparamに計上。"""
    row_idx = _const("ri", np.repeat(np.arange(H), k).astype(np.int64))
    col_idx = _const("ci", np.repeat(np.arange(W), k).astype(np.int64))
    g1 = helper.make_node("Gather", ["in", "ri"], ["g1"], axis=0)   # 行拡大
    g2 = helper.make_node("Gather", ["g1", "ci"], ["out"], axis=1)  # 列拡大
    return _build([g1, g2], [row_idx, col_idx], "upscale")

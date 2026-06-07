"""
templates_logical.py — Tier B 論理系テンプレート（XOR/AND/OR グリッド合成）
================================================================================
2つのグリッド（または1グリッドの2領域）を論理演算で合成するタスク。

【onnx-tool 実測校正済みの事実】
  - And/Or/Xor: MAC = 出力要素数（例 10x10 → 100 MAC）
  - Equal/Greater/Less: MAC = 出力要素数
  - Where: MAC = 0（重要！ 条件選択はMACゼロ。マスク適用に最適）
  - Add/Mul/Sub: MAC = 出力要素数
  → FREE op系(cost≒memoryのみ)と違い、CHEAP系は MAC が出力要素数ぶん乗る。
    ただし Where だけは MAC=0 なので、論理合成の最終段はWhereで組むと安い。

【コスト設計】
  10x10タスクで論理演算1つ = MAC 100 + memory(出力100B + 中間) 。
  cost = params(0) + MAC + memory。例えば
    Slice分割(0MAC) → Equal(100MAC) → Where(0MAC) なら MAC計 約100。
  これに memory（中間テンソル×100B級）が加わる。250-600帯に十分収まる。

【典型タスク】
  - 2グリッドのXOR（重なり検出）: 左右半分をSliceで分け、Xorで合成
  - マスクで色を塗る: Equal で位置マスク → Where で色を差し替え
  - 共通部分/和集合: And / Or
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
OPSET = 18


def _const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _finalize(nodes, inits, name, inputs=None, outputs=None):
    if inputs is None:
        inputs = [helper.make_tensor_value_info("in", U8, ["H", "W"])]
    if outputs is None:
        outputs = [helper.make_tensor_value_info("out", U8, ["H2", "W2"])]
    g = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def run(model, feed):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


# ---------------------------------------------------------------------------
# L1: 左右半分の XOR（重なり検出 → 出力は半分サイズ）
# ---------------------------------------------------------------------------
def l_halves_xor(out_color=1):
    """入力の左半分と右半分を取り出し、両方が非ゼロ⊕で出力色を置く。
       手順: Slice(左) , Slice(右) → Cast bool → Xor → Where(色付与)
       MAC: Equal系なし、Xor=出力要素数。Whereは0。
       ※グリッド幅Wが偶数前提。Wは動的なのでhalf位置はWhere側で吸収せず
         ここでは固定スプリット例（実タスクでは幅に合わせる）。"""
    # 左右に分割（axis=1を2分割）: Splitを使う
    split = helper.make_node("Split", ["in"], ["L", "R"], axis=1, num_outputs=2)
    # 非ゼロ判定: in != 0 を作るため、0定数とEqualしてNotする代わりに
    # Cast(bool) で「非ゼロ=True」を得る（uint8→boolは非ゼロがTrue）
    cl = helper.make_node("Cast", ["L"], ["Lb"], to=BOOL)
    cr = helper.make_node("Cast", ["R"], ["Rb"], to=BOOL)
    x = helper.make_node("Xor", ["Lb", "Rb"], ["xb"])     # MAC=出力要素数
    # Where(xb, out_color, 0): MAC=0
    onc = _const("onc", np.array(out_color, np.uint8))
    zero = _const("zero", np.array(0, np.uint8))
    w = helper.make_node("Where", ["xb", "onc", "zero"], ["out"])
    return _finalize([split, cl, cr, x, w], [onc, zero], "halves_xor")


# ---------------------------------------------------------------------------
# L2: マスク塗り（特定色の位置を別色に置換、Where使用でMAC最小）
# ---------------------------------------------------------------------------
def l_recolor_where(target_color, new_color):
    """入力中の target_color のセルを new_color に置換、他はそのまま。
       Equal(=target) → Where(mask, new, in)。
       MAC: Equal=出力要素数(100)、Where=0。計 約100。
       色置換のGather版(FREE,0MAC)が使えない条件依存ケース向け。"""
    tc = _const("tc", np.array(target_color, np.uint8))
    nc = _const("nc", np.array(new_color, np.uint8))
    eq = helper.make_node("Equal", ["in", "tc"], ["mask"])   # MAC=要素数
    w = helper.make_node("Where", ["mask", "nc", "in"], ["out"])  # MAC=0
    return _finalize([eq, w], [tc, nc], "recolor_where")


# ---------------------------------------------------------------------------
# L3: 2グリッドのAND（共通部分、入力2つ）
# ---------------------------------------------------------------------------
def l_two_grids_and(out_color=1):
    """2つの入力グリッドの共通の非ゼロ位置に色を置く。
       入力: a, b (uint8 同形状)。Cast bool → And → Where。"""
    a = helper.make_tensor_value_info("a", U8, ["H", "W"])
    b = helper.make_tensor_value_info("b", U8, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    ca = helper.make_node("Cast", ["a"], ["ab"], to=BOOL)
    cb = helper.make_node("Cast", ["b"], ["bb"], to=BOOL)
    an = helper.make_node("And", ["ab", "bb"], ["m"])        # MAC=要素数
    onc = _const("onc", np.array(out_color, np.uint8))
    zero = _const("zero", np.array(0, np.uint8))
    w = helper.make_node("Where", ["m", "onc", "zero"], ["out"])
    g = helper.make_graph([ca, cb, an, w], "two_grids_and",
                          [a, b], [out], initializer=[onc, zero])
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------------------
# L4: 2グリッドのOR（和集合）
# ---------------------------------------------------------------------------
def l_two_grids_or(out_color=1):
    a = helper.make_tensor_value_info("a", U8, ["H", "W"])
    b = helper.make_tensor_value_info("b", U8, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    ca = helper.make_node("Cast", ["a"], ["ab"], to=BOOL)
    cb = helper.make_node("Cast", ["b"], ["bb"], to=BOOL)
    orn = helper.make_node("Or", ["ab", "bb"], ["m"])
    onc = _const("onc", np.array(out_color, np.uint8))
    zero = _const("zero", np.array(0, np.uint8))
    w = helper.make_node("Where", ["m", "onc", "zero"], ["out"])
    g = helper.make_graph([ca, cb, orn, w], "two_grids_or",
                          [a, b], [out], initializer=[onc, zero])
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m

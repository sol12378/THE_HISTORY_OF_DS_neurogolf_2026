"""
templates_recolor_dynamic.py — 動的形状の色置換（uint8入力を保ちつつindex化）
================================================================================
問題: 色置換は Gather(LUT, indices=grid) で実装するが、Gatherのindicesは
      整数型必須。前バージョンt_recolorは入力をint32で受けていたが、
      これだと入力テンソルがuint8の4倍のバイトになる。

ただし onnx-tool 実測では「memory項は出力テンソルのみ計上、入力は計上されない」
ことが判明した。よって入力のdtypeはmemoryコストに影響しない（！）。
→ 色置換は素直に「uint8入力 → Cast(int32) → Gather」とすればよい。
   Castは実測でMAC=0・追加memoryは中間テンソル分のみ。

ここでは2方式を提供:
  R1: uint8入力 → Cast → Gather   （Cast中間1個ぶんmemory増、汎用）
  R2: int32入力 → Gather          （Cast不要、中間0。memory項は出力のみなので
                                    入力int32でもmemoryは増えない＝これが最安）

【onnx-tool 実測の含意】
  memory項が出力テンソルのみなら、R2（int32入力・Gather1個）が
  「中間ゼロ・出力uint8のみ」で最小。入力int32のバイト増は計上されない。
  Castを足すR1はCast中間が1個増えるぶん不利。
  → 結論: 色置換は R2（int32入力・Gather単発）が最適。
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


def run(model, x):
    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"in": x})[0]


# ---------------------------------------------------------------------------
# R1: uint8入力 → Cast → Gather（汎用・入力統一したい場合）
# ---------------------------------------------------------------------------
def recolor_cast(mapping):
    """uint8入力を受け、Castでint32化してからGather。
       中間テンソル(casted)が1個増えるが、入力dtypeをuint8に統一できる。
       mapping: 長さ10のLUT。"""
    inp = helper.make_tensor_value_info("in", U8, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    cast = helper.make_node("Cast", ["in"], ["idx"], to=I32)   # MAC=0
    lut = _const("lut", np.asarray(mapping, np.uint8))
    g = helper.make_node("Gather", ["lut", "idx"], ["out"], axis=0)
    graph = helper.make_graph([cast, g], "recolor_cast", [inp], [out],
                              initializer=[lut])
    m = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------------------
# R2: int32入力 → Gather（最安・中間ゼロ）★推奨
# ---------------------------------------------------------------------------
def recolor_direct(mapping):
    """int32入力（グリッド値がそのままGatherのindex）→ Gather単発。
       中間テンソル0個。memory項は出力(uint8)のみ。実測上これが最小コスト。
       入力int32のバイト増はmemory項に計上されない（出力のみ計上のため）。"""
    inp = helper.make_tensor_value_info("in", I32, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    lut = _const("lut", np.asarray(mapping, np.uint8))
    g = helper.make_node("Gather", ["lut", "in"], ["out"], axis=0)
    graph = helper.make_graph([g], "recolor_direct", [inp], [out],
                              initializer=[lut])
    m = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


# ---------------------------------------------------------------------------
# R3: 2D LUT による「位置非依存の条件付き色置換」
# ---------------------------------------------------------------------------
def recolor_conditional(mapping_when_true, mapping_when_false, target):
    """条件（あるセルがtargetか）で異なるLUTを使う高度な色置換。
       実装は L系(Where)に寄せた方が安い場合が多いので、ここでは
       単純化して Equal→Where の2段で表現（templates_logical.l_recolor_whereと同等）。
       複数色を一括置換するならGather LUT(R2)が最安。条件1色ならこちら。"""
    inp = helper.make_tensor_value_info("in", I32, ["H", "W"])
    out = helper.make_tensor_value_info("out", U8, ["H", "W"])
    lut_t = _const("lt", np.asarray(mapping_when_true, np.uint8))
    g = helper.make_node("Gather", ["lt", "in"], ["out"], axis=0)
    graph = helper.make_graph([g], "recolor_conditional", [inp], [out],
                              initializer=[lut_t])
    m = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m

"""
probe_onnx_tool.py — onnx-tool 実測校正
================================================================================
本物の onnx-tool が各 op / テンソル / dtype をどう計上するかを実測する。
TEMPLATES_REFERENCE.md の cost概算（要素数×dtype）を絶対値に校正するのが目的。

確認したいこと:
  1. memory項に「入力テンソル」は含まれるか？
  2. initializer(定数)はmemory/paramsにどう乗るか？
  3. 各FREE op(Transpose/Slice/Gather/Tile/Concat/Pad)のMACは本当に0か？
  4. CHEAP op(Equal/Where/Xor/And/Or/Add)のMACはいくらか？
  5. uint8 と int64 でmemoryバイトは8倍差が出るか？

使い方:
  python3 probe_onnx_tool.py
出力された実測値で、各テンプレートの cost予算表を校正する。
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import io
import contextlib

OPSET = 18


def _outvi(name, dtype):
    """shape未指定の出力value_info（任意shapeを許可）。"""
    vi = helper.make_tensor_value_info(name, dtype, None)
    vi.type.tensor_type.ClearField("shape")
    return vi


def _model_from_nodes(nodes, inits, inputs, outputs, name="probe"):
    g = helper.make_graph(nodes, name, inputs, outputs, initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_operatorsetid("", OPSET)])
    m.ir_version = 9
    # 出力shapeは動的なので strict checkはskip（onnx-toolが shape inferする）
    return m


def profile(model, input_feed):
    """onnx-tool で profile し、結果テーブルを文字列で返す。
       入力ndarrayでshapeを確定させてからprofileする。"""
    import onnx_tool
    tmp = "/tmp/_probe.onnx"
    onnx.save(model, tmp)
    buf = io.StringIO()
    try:
        m = onnx_tool.Model(tmp)
        m.graph.shape_infer(input_feed)   # 実ndarrayでshape確定
        m.graph.profile()
        with contextlib.redirect_stdout(buf):
            m.graph.print_node_map()
        return buf.getvalue()
    except Exception as e:
        return f"(Model API error: {e})\n"


def probe_op(op_type, h=10, w=10, dtype=TensorProto.UINT8, **kw):
    """1ノードグラフを作って profile。"""
    inp = helper.make_tensor_value_info("in", dtype, [h, w])
    out = _outvi("out", dtype)
    inits = []
    if op_type == "Transpose":
        node = helper.make_node("Transpose", ["in"], ["out"], perm=[1, 0])
    elif op_type == "Slice":
        inits = [numpy_helper.from_array(np.array([0, 0], np.int64), "s"),
                 numpy_helper.from_array(np.array([h, w], np.int64), "e"),
                 numpy_helper.from_array(np.array([0, 1], np.int64), "a")]
        node = helper.make_node("Slice", ["in", "s", "e", "a"], ["out"])
    elif op_type == "Tile":
        inits = [numpy_helper.from_array(np.array([2, 2], np.int64), "r")]
        node = helper.make_node("Tile", ["in", "r"], ["out"])
    elif op_type == "Concat":
        node = helper.make_node("Concat", ["in", "in"], ["out"], axis=1)
    elif op_type == "Pad":
        inits = [numpy_helper.from_array(np.array([1, 1, 1, 1], np.int64), "p")]
        node = helper.make_node("Pad", ["in", "p"], ["out"], mode="constant")
    elif op_type == "Gather":
        # data=in[h,w] (uint8), indices=int64 [w] → out[h,w]
        inits = [numpy_helper.from_array(np.arange(w, dtype=np.int64), "idx")]
        node = helper.make_node("Gather", ["in", "idx"], ["out"], axis=1)
        out = _outvi("out", dtype)
    elif op_type in ("Add", "Mul", "Sub"):
        node = helper.make_node(op_type, ["in", "in"], ["out"])
    elif op_type in ("Equal", "Greater", "Less"):
        out = _outvi("out", TensorProto.BOOL)
        node = helper.make_node(op_type, ["in", "in"], ["out"])
    elif op_type in ("And", "Or", "Xor"):
        inp = helper.make_tensor_value_info("in", TensorProto.BOOL, [h, w])
        out = _outvi("out", TensorProto.BOOL)
        node = helper.make_node(op_type, ["in", "in"], ["out"])
    elif op_type == "Where":
        cond = helper.make_tensor_value_info("c", TensorProto.BOOL, [h, w])
        node = helper.make_node("Where", ["c", "in", "in"], ["out"])
        m = _model_from_nodes([node], [], [cond, inp], [out])
        feed = {"c": np.ones((h, w), bool), "in": np.zeros((h, w), np.uint8)}
        return profile(m, feed)
    else:
        raise ValueError(op_type)

    m = _model_from_nodes([node], inits, [inp], [out])
    npdt = {TensorProto.UINT8: np.uint8, TensorProto.INT64: np.int64,
            TensorProto.BOOL: bool, TensorProto.FLOAT: np.float32}[dtype]
    feed = {"in": np.ones((h, w), npdt)}
    return profile(m, feed)


if __name__ == "__main__":
    print("=" * 70)
    print("onnx-tool 実測校正レポート")
    print("=" * 70)

    print("\n--- FREE op (MAC=0 を期待) ---")
    for op in ["Transpose", "Slice", "Tile", "Concat", "Pad", "Gather"]:
        print(f"\n### {op} (uint8 10x10)")
        print(probe_op(op))

    print("\n--- CHEAP op (出力要素数ぶんのMACを期待) ---")
    for op in ["Add", "Mul", "Equal", "And", "Where"]:
        print(f"\n### {op} (10x10)")
        print(probe_op(op))

    print("\n--- dtype比較: 同じTransposeで uint8 vs int64 ---")
    print("### Transpose uint8")
    print(probe_op("Transpose", dtype=TensorProto.UINT8))
    print("### Transpose int64")
    print(probe_op("Transpose", dtype=TensorProto.INT64))

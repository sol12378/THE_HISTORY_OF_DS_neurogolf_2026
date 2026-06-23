"""Phase-2 P2-3 step 1: test float16 conversion (keep I/O float32) on a few high-cost
union artifacts THROUGH the real gate (evaluate_candidate: static + full-arc + rescore).
Accept only if full-arc passes AND cost strictly drops. Zero risk: rejects are harmless."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPROOT = HERE.parent
sys.path.insert(0, str(EXPROOT))

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto
import phase1_rewrite_utils as P
import neurogolf_calib as C

CLIMB = EXPROOT / "exp337_autonomous_climb"
UNION = EXPROOT / "exp336_public_lb7117_task_best_union" / "submission.zip"

TEST_TASKS = [187, 2, 209, 255, 366, 80]
raws = C._read_zip(UNION)
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
utils = P.load_neurogolf_utils()

FLOAT = TensorProto.FLOAT
FLOAT16 = TensorProto.FLOAT16


def to_fp16_occ(raw: bytes) -> bytes:
    """Manual float16 conversion: all float32 initializers/constants/intermediates ->
    float16, graph input/output kept float32 and bridged by Cast nodes. value_info is
    cleared and re-inferred by the scorer. Anything ORT cannot run in fp16 is caught
    by the gate (rejected, never accepted)."""
    model = onnx.load_model_from_string(raw)
    g = model.graph
    in_name, out_name = g.input[0].name, g.output[0].name
    for init in g.initializer:
        if init.data_type == FLOAT:
            arr = numpy_helper.to_array(init).astype(np.float16)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for node in g.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == FLOAT:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FLOAT:
                    attr.i = FLOAT16
    in16, out16 = "__in_f16", "__out_f16"
    for node in g.node:
        for i, x in enumerate(node.input):
            if x == in_name:
                node.input[i] = in16
        for j, y in enumerate(node.output):
            if y == out_name:
                node.output[j] = out16
        for i, x in enumerate(node.input):
            if x == out_name:
                node.input[i] = out16
    cast_in = onnx.helper.make_node("Cast", [in_name], [in16], to=FLOAT16, name="__cast_in")
    cast_out = onnx.helper.make_node("Cast", [out16], [out_name], to=FLOAT, name="__cast_out")
    new_nodes = [cast_in] + list(g.node) + [cast_out]
    del g.node[:]
    g.node.extend(new_nodes)
    del g.value_info[:]
    return model.SerializeToString()


print(f"{'tid':>4} {'base_cost':>9} {'fp16_cost':>9} {'val_status':>14} {'verdict'}")
for tid in TEST_TASKS:
    raw = raws[tid]
    try:
        fp16_raw = to_fp16_occ(raw)
    except Exception as exc:
        print(f"{tid:>4} {costs[tid]:>9}  convert failed: {str(exc)[:80]}")
        continue
    cand = P.Candidate(tid, "fp16_layerC", "", fp16_raw, "generated", "float16 internal")
    base = P.BaseTask(tid, costs[tid], P.point(costs[tid]), "union", "union", "", raw)
    ev, acc = P.evaluate_candidate(utils, cand, base, -1, CLIMB)
    cc = ev.candidate_cost if isinstance(ev.candidate_cost, int) else "n/a"
    verdict = "ACCEPT (cheaper+valid)" if acc is not None else f"reject: {ev.reason[:40]}"
    print(f"{tid:>4} {costs[tid]:>9} {str(cc):>9} {ev.validation_status:>14} {verdict}")

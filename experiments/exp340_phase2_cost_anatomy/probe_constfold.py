"""Measure constant-fold yield: for top-cost tasks, how many intermediate tensors are
input-INDEPENDENT (identical across different inputs) and their float32 element mass.
Baking a constant float32 intermediate of N elems: memory -4N bytes, params +N elems
=> cost -3N. Decides whether a fold pass is worth building."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPROOT = HERE.parent
sys.path.insert(0, str(EXPROOT))

import numpy as np
import onnx
import onnxruntime as ort
import phase1_rewrite_utils as P
import neurogolf_calib as C

CLIMB = EXPROOT / "exp337_autonomous_climb"
UNION = EXPROOT / "exp336_public_lb7117_task_best_union" / "submission.zip"
TOP = [187, 233, 18, 286, 209, 205, 118, 191, 255, 2, 349, 367, 202, 219]
raws = C._read_zip(UNION)
led = json.loads((CLIMB / "baseline_ledger.json").read_text(encoding="utf-8"))
costs = {int(k): int(v) for k, v in led["costs"].items()}
utils = P.load_neurogolf_utils()


def constant_mass(tid):
    raw = raws[tid]
    model = onnx.load_model_from_string(raw)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        inferred = model
    vi_type = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info
               if v.type.HasField("tensor_type")}
    inter = [o for n in model.graph.node for o in n.output if o and o not in ("input", "output")]
    inter = [t for t in inter if t in vi_type]  # only those with a known type
    probe = onnx.load_model_from_string(raw)
    pg = probe.graph
    have = {o.name for o in pg.output}
    for t in inter:
        if t not in have:
            pg.output.append(onnx.helper.make_tensor_value_info(t, vi_type[t], None))
    try:
        sess = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    except Exception as exc:
        return None, f"probe build failed: {str(exc)[:50]}"
    out_names = [o.name for o in pg.output if o.name in inter]
    # three diverse real inputs
    exs = P.examples_for(P.load_task(tid), 4)
    inputs = [utils.convert_to_numpy(e)["input"] for e in exs[:3] if utils.convert_to_numpy(e)]
    if len(inputs) < 2:
        return None, "not enough inputs"
    runs = []
    for inp in inputs:
        try:
            runs.append(dict(zip(out_names, sess.run(out_names, {"input": inp}))))
        except Exception as exc:
            return None, f"run failed: {str(exc)[:50]}"
    const_elems = 0
    const_f32_elems = 0
    n_const = 0
    for t in out_names:
        vals = [r[t] for r in runs]
        if all(np.array_equal(vals[0], v) for v in vals[1:]):
            n_const += 1
            const_elems += vals[0].size
            if vi_type[t] == onnx.TensorProto.FLOAT:
                const_f32_elems += vals[0].size
    return (n_const, len(out_names), const_f32_elems), "ok"


print(f"{'tid':>4} {'cost':>7} {'#const':>7}/{'#inter':<6} {'const_f32_elems':>15} {'est_cost_drop(-3N)':>18}")
for tid in TOP:
    res, reason = constant_mass(tid)
    if res is None:
        print(f"{tid:>4} {costs[tid]:>7}  {reason}")
        continue
    n_const, n_inter, f32e = res
    print(f"{tid:>4} {costs[tid]:>7} {n_const:>7}/{n_inter:<6} {f32e:>15} {-3 * f32e:>18}")

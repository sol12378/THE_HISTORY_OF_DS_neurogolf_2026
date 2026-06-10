from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, score_model, validate_examples


EXP_DIR = ROOT / "experiments" / "exp194_final_bool_output_keep_name_probe"
INPUTS = {
    206: ROOT / "experiments" / "exp161_where_dtype_rewrite_inspection" / "task206.onnx",
    328: ROOT / "experiments" / "exp161_where_dtype_rewrite_inspection" / "task328.onnx",
}


def remove_final_cast_keep_output_name(raw: bytes) -> bytes:
    model = onnx.load_from_string(raw)
    graph = model.graph
    output_name = graph.output[0].name
    final_cast = None
    for node in graph.node:
        if node.op_type == "Cast" and output_name in node.output:
            final_cast = node
            break
    if final_cast is None:
        raise ValueError("final Cast to output not found")
    bool_name = final_cast.input[0]
    producers = [node for node in graph.node if bool_name in node.output]
    if len(producers) != 1:
        raise ValueError(f"expected one producer for {bool_name}, found {len(producers)}")
    producer = producers[0]
    for i, out in enumerate(producer.output):
        if out == bool_name:
            producer.output[i] = output_name
    for node in graph.node:
        if node is producer or node is final_cast:
            continue
        for i, inp in enumerate(node.input):
            if inp == bool_name:
                node.input[i] = output_name
    graph.node.remove(final_cast)
    graph.output[0].type.tensor_type.elem_type = TensorProto.BOOL
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def probe_task(utils: Any, task_id: int, path: Path) -> dict[str, Any]:
    base_raw = path.read_bytes()
    base_memory, base_params, base_reason = score_model(utils, base_raw, task_id, f"exp194_task{task_id:03d}_base", EXP_DIR)
    try:
        cand_raw = remove_final_cast_keep_output_name(base_raw)
    except Exception as exc:
        return {"task_id": task_id, "status": "rewrite_failed", "reason": str(exc)}
    out_path = EXP_DIR / f"task{task_id:03d}_final_bool_keep_output.onnx"
    out_path.write_bytes(cand_raw)
    ok, reason, passed, failed = validate_examples(utils, cand_raw, task_id, arc_gen_sample=-1)
    memory, params, score_reason = score_model(utils, cand_raw, task_id, f"exp194_task{task_id:03d}_final_bool_keep_output", EXP_DIR)
    base_cost = int(base_memory + base_params) if base_memory is not None and base_params is not None else None
    cand_cost = int(memory + params) if memory is not None and params is not None else None
    return {
        "task_id": task_id,
        "status": "valid_cost_improved" if ok and cand_cost is not None and base_cost is not None and cand_cost < base_cost else "not_accepted",
        "validation": {"ok": ok, "reason": reason, "passed": passed, "failed": failed, "status": f"{passed}_pass_{failed}_fail"},
        "base_cost": base_cost,
        "candidate_cost": cand_cost,
        "cost_delta": (base_cost - cand_cost) if base_cost is not None and cand_cost is not None else None,
        "base_score_reason": base_reason,
        "candidate_score_reason": score_reason,
        "candidate_path": str(out_path.relative_to(ROOT)),
    }


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    rows = [probe_task(utils, task_id, path) for task_id, path in INPUTS.items()]
    accepted = [r for r in rows if r["status"] == "valid_cost_improved"]
    result = {
        "exp_id": "exp194_final_bool_output_keep_name_probe",
        "date": "2026-06-10",
        "status": "probe_complete",
        "purpose": "Retry final FLOAT Cast removal while preserving output tensor name.",
        "rows": rows,
        "accepted_count": len(accepted),
        "decision": "Adopt only if BOOL output validates and official score reports lower cost.",
        "elapsed_s": round(time.time() - t0, 3),
        "leakage_risk": "low: graph dtype/cost probe only.",
        "overfitting_risk": "low-to-medium: output dtype acceptance may differ from Kaggle packaging expectations.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "# exp194_final_bool_output_keep_name_probe\n\n"
        "## 目的\n\n最終FLOAT Cast除去をoutput名維持で再試行する。\n\n"
        "## 結果\n\n"
        f"- accepted_count: `{len(accepted)}`\n\n"
        "```json\n" + json.dumps(rows, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

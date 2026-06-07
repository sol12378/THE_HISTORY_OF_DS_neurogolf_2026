from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import date
from typing import Any

import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_base_tasks, load_neurogolf_utils, load_task, sha256


EXP_ID = "exp045_task031_artifact_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
TARGET_TASK = 31


def profile_raw(utils: Any, raw: bytes) -> tuple[int | None, int | None, pathlib.Path | None, str]:
    try:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = str(EXP_DIR / f"profile_task{TARGET_TASK:03d}")
        session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
        benchmark = utils.convert_to_numpy(examples_for(load_task(TARGET_TASK), 1)[0])
        utils.run_network(session, benchmark["input"])
        trace_path = pathlib.Path(session.end_profiling())
        memory, params = utils.score_network(onnx.load_model_from_string(raw), str(trace_path))
        return memory, params, trace_path, "ok"
    except Exception as exc:
        return None, None, None, f"profile failed: {str(exc)[:180]}"


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)[TARGET_TASK]
    model = onnx.load_model_from_string(base.raw)
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError("sanitize failed")
    raw = sanitized.SerializeToString()
    memory, params, trace_path, reason = profile_raw(utils, raw)

    node_rows: list[dict[str, Any]] = []
    op_counts = Counter()
    for idx, node in enumerate(sanitized.graph.node):
        op_counts[node.op_type] += 1
        node_rows.append(
            {
                "idx": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": "|".join(node.input),
                "outputs": "|".join(node.output),
                "attr_count": len(node.attribute),
            }
        )
    with (EXP_DIR / "node_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "name", "op_type", "inputs", "outputs", "attr_count"])
        writer.writeheader()
        writer.writerows(node_rows)

    init_rows: list[dict[str, Any]] = []
    for init in sanitized.graph.initializer:
        size = 1
        for dim in init.dims:
            size *= int(dim)
        init_rows.append({"name": init.name, "dims": "x".join(str(x) for x in init.dims), "elem_count": size, "data_type": init.data_type})
    init_rows = sorted(init_rows, key=lambda row: (-int(row["elem_count"]), row["name"]))
    with (EXP_DIR / "initializer_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "dims", "elem_count", "data_type"])
        writer.writeheader()
        writer.writerows(init_rows)

    trace_summary: list[dict[str, Any]] = []
    if trace_path and trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for event in trace:
            if event.get("cat") != "Node":
                continue
            args = event.get("args", {})
            trace_summary.append(
                {
                    "name": str(event.get("name", "")).replace("_kernel_time", ""),
                    "dur": event.get("dur", ""),
                    "output_type_shape": json.dumps(args.get("output_type_shape", []), ensure_ascii=False),
                }
            )
        with (EXP_DIR / "trace_node_shapes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "dur", "output_type_shape"])
            writer.writeheader()
            writer.writerows(trace_summary)
        trace_path.unlink(missing_ok=True)

    high_level = {
        "node_count": len(sanitized.graph.node),
        "initializer_count": len(sanitized.graph.initializer),
        "initializer_elem_total": sum(int(row["elem_count"]) for row in init_rows),
        "largest_initializers": init_rows[:10],
        "op_counts": dict(op_counts),
        "uses_dynamic_index_ops": any(op in op_counts for op in ["GatherND", "ScatterND", "ArgMax"]),
        "uses_conv": "Conv" in op_counts,
        "uses_where": "Where" in op_counts,
    }
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "diagnostic_complete" if memory is not None and params is not None else "failed",
        "target_task": TARGET_TASK,
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "artifact_cost_from_manifest": base.cost,
        "profile_memory": memory,
        "profile_params": params,
        "profile_cost": None if memory is None or params is None else memory + params,
        "profile_reason": reason,
        "sha256": sha256(raw),
        "high_level": high_level,
        "runtime_seconds": time.time() - started,
        "decision": "Use this artifact profile to design a smaller bbox/object compiler; do not use full-grid GatherND.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    notes = f"""# {EXP_ID}

## 目的

exp044でfull-grid `GatherND` bbox loweringが破綻したため、task031の既存artifactがcost {base.cost} に収まっている構造を読む。

## 結果

- status: {result["status"]}
- manifest cost: {base.cost}
- profiled cost: {result["profile_cost"]}
- node count: {high_level["node_count"]}
- initializer count: {high_level["initializer_count"]}
- op counts: {dict(op_counts)}
- largest initializers: {init_rows[:5]}

## 解釈

既存artifactのop構成とinitializerサイズを、次のbbox/object compiler設計の制約として使う。特に `GatherND/ScatterND/ArgMax/Where/Conv` の有無と大きなinitializerの有無を見る。

## 次

`node_inventory.csv` と `trace_node_shapes.csv` から、bbox cropがstatic lookup寄りなのか、小さいmask演算寄りなのかを判断する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()

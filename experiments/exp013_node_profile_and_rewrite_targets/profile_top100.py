from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time
from collections import Counter
from typing import Any

import onnx
import onnxruntime as ort

sys_path_root = pathlib.Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(sys_path_root))
from experiments.phase1_rewrite_utils import ROOT, examples_for, load_base_tasks, load_neurogolf_utils, load_task, top_task_ids


EXP_DIR = ROOT / "experiments" / "exp013_node_profile_and_rewrite_targets"
TARGET_OPS = {"Where", "MaxPool", "MatMul", "Tile", "Slice", "Gather", "ScatterElements", "ScatterND", "QLinearMatMul", "Conv"}


def profile_trace(utils: Any, raw: bytes, task_id: int) -> dict[str, int]:
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(EXP_DIR / f"profile_task{task_id:03d}")
    session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
    benchmark = utils.convert_to_numpy(examples_for(load_task(task_id), 1)[0])
    if benchmark is not None:
        utils.run_network(session, benchmark["input"])
    trace = pathlib.Path(session.end_profiling())
    node_seen: dict[str, int] = {}
    data = json.loads(trace.read_text(encoding="utf-8"))
    trace.unlink(missing_ok=True)
    for event in data:
        if event.get("cat") != "Node" or "args" not in event:
            continue
        name = str(event.get("name", "")).replace("_kernel_time", "")
        if name:
            node_seen[name] = node_seen.get(name, 0) + 1
    return node_seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    task_ids = top_task_ids(base, args.top_k)
    node_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    started = time.time()

    for idx, task_id in enumerate(task_ids, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"Profiling {idx}/{len(task_ids)} task{task_id:03d}", flush=True)
        task = base[task_id]
        model = onnx.load_model_from_string(task.raw)
        trace_hits = profile_trace(utils, task.raw, task_id)
        op_counts = Counter(node.op_type for node in model.graph.node)
        target_count = sum(count for op, count in op_counts.items() if op in TARGET_OPS)
        for node_index, node in enumerate(model.graph.node):
            row = {
                "task_id": task_id,
                "route": task.route,
                "baseline_cost": task.cost,
                "node_index": node_index,
                "node_name": node.name,
                "op_type": node.op_type,
                "input_count": len(node.input),
                "output_count": len(node.output),
                "profile_hits": trace_hits.get(node.name, 0),
                "is_target_op": node.op_type in TARGET_OPS,
            }
            node_rows.append(row)
        target_rows.append(
            {
                "task_id": task_id,
                "route": task.route,
                "baseline_cost": task.cost,
                "node_count": len(model.graph.node),
                "initializer_count": len(model.graph.initializer),
                "target_op_node_count": target_count,
                "dominant_ops": ";".join(f"{op}:{count}" for op, count in op_counts.most_common(8)),
                "recommended_bank": "crop_resize" if task.route == "crop_or_resize" else "sparse_object",
                "priority": idx,
            }
        )

    with (EXP_DIR / "node_profile.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "route", "baseline_cost", "node_index", "node_name", "op_type", "input_count", "output_count", "profile_hits", "is_target_op"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(node_rows)

    with (EXP_DIR / "rewrite_target_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "route", "baseline_cost", "node_count", "initializer_count", "target_op_node_count", "dominant_ops", "recommended_bank", "priority"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(target_rows)

    result = {
        "exp_id": "exp013_node_profile_and_rewrite_targets",
        "date": "2026-06-05",
        "status": "profile_complete",
        "top_k": args.top_k,
        "node_rows": len(node_rows),
        "route_counts": dict(Counter(row["route"] for row in target_rows)),
        "runtime_seconds": time.time() - started,
        "artifacts": {
            "node_profile": str((EXP_DIR / "node_profile.csv").relative_to(ROOT)),
            "rewrite_target_manifest": str((EXP_DIR / "rewrite_target_manifest.csv").relative_to(ROOT)),
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(
        "# exp013_node_profile_and_rewrite_targets notes\n\n"
        "top100 taskのONNX node構成を集計し、template bankで置換すべきroute/op patternを整理した。\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

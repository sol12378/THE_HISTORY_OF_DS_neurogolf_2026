from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date
from typing import Any

import onnx
from onnx import helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_neurogolf_utils,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp112_task185_intermediate_output_extraction"
EXP_DIR = ROOT / "experiments" / EXP_ID
SEED_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
CURRENT_BUNDLE = ROOT / "experiments" / "exp111_focused_surgery_second_pass" / "submission.zip"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
TASK_ID = 185
CAMPAIGN_INDEX = 14


def load_current_tasks() -> dict[int, BaseTask]:
    rows: dict[int, dict[str, str]] = {}
    with (SEED_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(CURRENT_BUNDLE) as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    tasks: dict[int, BaseTask] = {}
    for task_id, row in rows.items():
        tasks[task_id] = BaseTask(
            task_id=task_id,
            cost=int(float(row["cost"])),
            points=float(row["local_points"]),
            source=row["source"],
            template_name=row["template_name"],
            route=row.get("route", ""),
            raw=raws[task_id],
        )
    return tasks


def tensor_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes: dict[str, list[int]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                ok = False
                break
            dims.append(int(dim.dim_value))
        if ok:
            shapes[value.name] = dims
    return shapes


def initializer_names(model: onnx.ModelProto) -> set[str]:
    return {init.name for init in model.graph.initializer}


def keep_ancestor_subgraph(model: onnx.ModelProto, output_tensor: str) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)

    producer: dict[str, int] = {}
    for idx, node in enumerate(out.graph.node):
        for name in node.output:
            if name:
                producer[name] = idx

    init_names = initializer_names(out)
    needed_values = {output_tensor}
    needed_nodes: set[int] = set()
    stack = [output_tensor]
    while stack:
        value = stack.pop()
        idx = producer.get(value)
        if idx is None or idx in needed_nodes:
            continue
        needed_nodes.add(idx)
        node = out.graph.node[idx]
        for inp in node.input:
            if not inp or inp in init_names:
                continue
            if inp not in needed_values:
                needed_values.add(inp)
                stack.append(inp)

    nodes = [node for idx, node in enumerate(out.graph.node) if idx in needed_nodes]
    del out.graph.node[:]
    out.graph.node.extend(nodes)

    identity = helper.make_node("Identity", [output_tensor], ["output"], name="output")
    out.graph.node.extend([identity])
    out.graph.output[0].name = "output"

    used_inputs = set()
    for node in out.graph.node:
        used_inputs.update(inp for inp in node.input if inp)
    kept_inits = [init for init in out.graph.initializer if init.name in used_inputs]
    del out.graph.initializer[:]
    out.graph.initializer.extend(kept_inits)

    return out


def output_shape_candidates(base: BaseTask, limit: int = 120) -> tuple[list[Candidate], list[dict[str, Any]], list[dict[str, Any]]]:
    model = onnx.load_model_from_string(base.raw)
    shapes = tensor_shapes(model)
    rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    candidates: list[Candidate] = []
    graph_output = model.graph.output[0].name
    target_shape = shapes.get(graph_output, [1, 10, 30, 30])

    for idx, node in enumerate(model.graph.node):
        for out_idx, name in enumerate(node.output):
            if not name or name == graph_output:
                continue
            shape = shapes.get(name)
            shape_rows.append(
                {
                    "node_idx": idx,
                    "op_type": node.op_type,
                    "output_idx": out_idx,
                    "tensor": name,
                    "shape": "" if shape is None else "x".join(str(x) for x in shape),
                }
            )
            if shape != target_shape:
                continue
            candidate_model = keep_ancestor_subgraph(model, name)
            raw = candidate_model.SerializeToString()
            rows.append(
                {
                    "node_idx": idx,
                    "op_type": node.op_type,
                    "output_idx": out_idx,
                    "tensor": name,
                    "node_count": len(candidate_model.graph.node),
                    "initializer_count": len(candidate_model.graph.initializer),
                    "file_bytes": len(raw),
                    "sha256": sha256(raw),
                }
            )
            candidates.append(
                Candidate(
                    TASK_ID,
                    f"extract_output_{idx:03d}_{node.op_type}_out{out_idx}",
                    base.route,
                    raw,
                    "generated",
                    f"use intermediate tensor {name} from node {idx} ({node.op_type}) as graph output",
                )
            )
            if len(candidates) >= limit:
                return candidates, rows, shape_rows
    return candidates, rows, shape_rows


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    base = base_tasks[TASK_ID]

    candidates, profile_rows, shape_rows = output_shape_candidates(base)
    evals: list[CandidateEval] = []
    best_eval: CandidateEval | None = None
    accepted: dict[int, bytes] = {}

    for candidate in candidates:
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved" and (best_eval is None or int(ev.candidate_cost) < int(best_eval.candidate_cost)):
            best_eval = ev
            accepted[TASK_ID] = raw

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    with (EXP_DIR / "intermediate_output_profile.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["node_idx", "op_type", "output_idx", "tensor", "node_count", "initializer_count", "file_bytes", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profile_rows)

    with (EXP_DIR / "shape_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["node_idx", "op_type", "output_idx", "tensor", "shape"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(shape_rows)

    shape_counts: dict[str, int] = {}
    for row in shape_rows:
        key = str(row["shape"])
        shape_counts[key] = shape_counts.get(key, 0) + 1
    top_shape_counts = [
        {"shape": shape, "count": count}
        for shape, count in sorted(shape_counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]

    status_counts: dict[str, int] = {}
    validation_counts: dict[str, int] = {}
    for ev in evals:
        status_counts[ev.status] = status_counts.get(ev.status, 0) + 1
        validation_counts[ev.validation_status] = validation_counts.get(ev.validation_status, 0) + 1

    delta = 0.0 if best_eval is None else float(best_eval.candidate_points) - float(best_eval.baseline_points)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": CAMPAIGN_INDEX,
        "task_id": TASK_ID,
        "hypothesis": "A task185 intermediate tensor may already equal the final padded one-hot output; extracting it can prune a large suffix.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "candidate_tensor_count": len(candidates),
        "target_output_shape": "x".join(str(x) for x in tensor_shapes(onnx.load_model_from_string(base.raw)).get(onnx.load_model_from_string(base.raw).graph.output[0].name, [])),
        "output_shape_tensor_count": len(profile_rows),
        "all_intermediate_tensor_count": len(shape_rows),
        "top_shape_counts": top_shape_counts,
        "evaluated_candidate_count": len(evals),
        "status_counts": status_counts,
        "validation_counts": validation_counts,
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "best_eval": asdict(best_eval) if best_eval is not None else None,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "If no intermediate output validates with lower cost, task185 needs a fresh fused lowering rather than suffix extraction.",
        "leakage_risk": "low: graph-only subgraph extraction, full-validation gated.",
        "overfitting_risk": "low-to-medium: accepted extraction would still need LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## Hypothesis

task185の既存artifact内部に、最終出力と同じshapeのone-hot tensorが早い段階で存在するなら、そのtensorを `Identity -> output` にして下流subgraphを丸ごとpruneできる。

## Result

- candidate tensors: `{len(candidates)}`
- all intermediate tensors: `{len(shape_rows)}`
- top shape counts: `{top_shape_counts}`
- status counts: `{status_counts}`
- validation counts: `{validation_counts}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Interpretation

中間output抽出が通れば、局所bypassより大きいsubgraph extractionとして採用する。通らない場合、task185 artifactは最終段まで必要な変換を分散しており、既存graphのsuffix cutではなくfresh fused loweringが必要。

## Risk

- leakage risk: low。既存graphの中間tensorを使うだけで、output lookupは使わない。
- overfitting risk: low-to-medium。acceptedが出た場合はLB calibration候補。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

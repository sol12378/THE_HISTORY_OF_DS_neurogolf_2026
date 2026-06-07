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
from onnx import TensorProto, helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp121_task366_cast_suffix_extraction"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_BUNDLE = ROOT / "experiments" / "exp111_focused_surgery_second_pass" / "submission.zip"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.93615685804
TASK_ID = 366
CAMPAIGN_INDEX = 23


def load_current_tasks() -> dict[int, BaseTask]:
    base_tasks = load_base_tasks()
    with zipfile.ZipFile(CURRENT_BUNDLE) as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    return {
        task_id: BaseTask(
            task_id=task_id,
            cost=task.cost,
            points=task.points,
            source=task.source,
            template_name=task.template_name,
            route=task.route,
            raw=raws[task_id],
        )
        for task_id, task in base_tasks.items()
    }


def tensor_info(model: onnx.ModelProto) -> tuple[dict[str, list[int]], dict[str, int]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, int] = {}
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
            dtypes[value.name] = int(tensor_type.elem_type)
    return shapes, dtypes


def keep_ancestor_subgraph(model: onnx.ModelProto, output_tensor: str, suffix: str, output_dtype: int) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    producer: dict[str, int] = {}
    for idx, node in enumerate(out.graph.node):
        for name in node.output:
            if name:
                producer[name] = idx
    init_names = {init.name for init in out.graph.initializer}
    needed_nodes: set[int] = set()
    stack = [output_tensor]
    while stack:
        value = stack.pop()
        idx = producer.get(value)
        if idx is None or idx in needed_nodes:
            continue
        needed_nodes.add(idx)
        for inp in out.graph.node[idx].input:
            if inp and inp not in init_names:
                stack.append(inp)
    nodes = [node for idx, node in enumerate(out.graph.node) if idx in needed_nodes]
    del out.graph.node[:]
    out.graph.node.extend(nodes)
    if suffix == "identity":
        out.graph.node.extend([helper.make_node("Identity", [output_tensor], ["output"], name="output")])
    elif suffix == "cast_output_dtype":
        out.graph.node.extend([helper.make_node("Cast", [output_tensor], ["output"], name="output", to=output_dtype)])
    else:
        raise ValueError(suffix)
    out.graph.output[0].name = "output"
    used_inputs = {inp for node in out.graph.node for inp in node.input if inp}
    kept_inits = [init for init in out.graph.initializer if init.name in used_inputs]
    del out.graph.initializer[:]
    out.graph.initializer.extend(kept_inits)
    return out


def build_candidates(base: BaseTask, limit: int = 240) -> tuple[list[Candidate], list[dict[str, Any]]]:
    model = onnx.load_model_from_string(base.raw)
    shapes, dtypes = tensor_info(model)
    graph_output = model.graph.output[0].name
    target_shape = shapes.get(graph_output, [1, 10, 30, 30])
    output_dtype = dtypes.get(graph_output, TensorProto.FLOAT)
    candidates: list[Candidate] = []
    profiles: list[dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        for out_idx, name in enumerate(node.output):
            if not name or name == graph_output:
                continue
            shape = shapes.get(name)
            if shape != target_shape:
                continue
            dtype = dtypes.get(name, -1)
            suffixes = ["identity"] if dtype == output_dtype else ["cast_output_dtype"]
            for suffix in suffixes:
                candidate_model = keep_ancestor_subgraph(model, name, suffix, output_dtype)
                raw = candidate_model.SerializeToString()
                profiles.append(
                    {
                        "node_idx": idx,
                        "op_type": node.op_type,
                        "output_idx": out_idx,
                        "tensor": name,
                        "dtype": dtype,
                        "suffix": suffix,
                        "node_count": len(candidate_model.graph.node),
                        "initializer_count": len(candidate_model.graph.initializer),
                        "file_bytes": len(raw),
                        "sha256": sha256(raw),
                    }
                )
                candidates.append(
                    Candidate(
                        TASK_ID,
                        f"{suffix}_{idx:03d}_{node.op_type}_out{out_idx}",
                        base.route,
                        raw,
                        "generated",
                        f"use intermediate tensor {name} from node {idx} ({node.op_type}) with suffix={suffix}",
                    )
                )
                if len(candidates) >= limit:
                    return candidates, profiles
    return candidates, profiles


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    tasks = load_current_tasks()
    base = tasks[TASK_ID]
    candidates, profiles = build_candidates(base)

    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval: CandidateEval | None = None
    for candidate in candidates:
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved" and (best_eval is None or int(ev.candidate_cost) < int(best_eval.candidate_cost)):
            best_eval = ev
            accepted[TASK_ID] = raw

    final_raw = {task_id: task.raw for task_id, task in tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])
    with (EXP_DIR / "cast_suffix_profile.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["node_idx", "op_type", "output_idx", "tensor", "dtype", "suffix", "node_count", "initializer_count", "file_bytes", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profiles)

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
        "hypothesis": "Task366 final-shape mask/type intermediates may become valid cheaper outputs with only a Cast-to-float suffix.",
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "candidate_count": len(candidates),
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
        "decision": "If Cast suffix fails, task366 artifact masks are not directly output-compatible and this lane should pivot.",
        "leakage_risk": "low: graph-only extraction, full-validation gated.",
        "overfitting_risk": "low-to-medium: accepted extraction would still need LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

task366のfinal-shape mask/type intermediateは、`Cast -> output` だけを足せば正しいfloat outputとして使える可能性がある。

## Result

- candidates: `{len(candidates)}`
- status counts: `{status_counts}`
- validation counts: `{validation_counts}`
- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`

## Interpretation

Cast suffixで通れば、task366既存artifactの大きなsuffixを削れる。通らなければ、mask intermediateは最終outputと意味的に違うため、このlaneはpivotする。

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import onnx


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    candidate_fields,
    evaluate_candidate,
    load_neurogolf_utils,
    load_routes,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b014_task037_fullarc_graph_surgery"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
TASK_ID = 37
STRICT_SEED_SCORE = 6282.230228


@dataclass(frozen=True)
class ProfileSummary:
    task_id: int
    node_count: int
    initializer_count: int
    file_bytes: int
    op_counts: str
    candidate_count: int


def load_strict_seed_tasks() -> dict[int, BaseTask]:
    routes = load_routes()
    rows: dict[int, dict[str, str]] = {}
    with (BASE_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    out: dict[int, BaseTask] = {}
    for task_id, row in rows.items():
        out[task_id] = BaseTask(
            task_id=task_id,
            cost=int(float(row["new_cost"])),
            points=float(row["new_points"]),
            source=row["source"],
            template_name="strict_seed",
            route=routes.get(task_id, ""),
            raw=raws[task_id],
        )
    return out


def replace_all_inputs(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == old:
                node.input[idx] = new
    for output in model.graph.output:
        if output.name == old:
            output.name = new


def remove_node_by_output(model: onnx.ModelProto, output_name: str) -> None:
    kept = [node for node in model.graph.node if output_name not in set(node.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)


def remove_unused_initializers(model: onnx.ModelProto) -> None:
    used = set()
    for node in model.graph.node:
        used.update(node.input)
    kept = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)


def bypass_candidates(task: BaseTask) -> Iterable[Candidate]:
    raw = task.raw
    model = onnx.load_model_from_string(raw)
    bypass_ops = {"And", "Or", "Where", "Add", "Mul", "Sub", "Cast", "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Greater", "Less", "Equal"}
    for idx, node in enumerate(model.graph.node):
        if len(node.output) != 1 or node.op_type not in bypass_ops:
            continue
        output_name = node.output[0]
        for input_idx, input_name in enumerate(node.input):
            if not input_name:
                continue
            candidate_model = onnx.load_model_from_string(raw)
            replace_all_inputs(candidate_model, output_name, input_name)
            remove_node_by_output(candidate_model, output_name)
            remove_unused_initializers(candidate_model)
            yield Candidate(
                task.task_id,
                f"task037_bypass_{node.op_type}_node{idx}_to_input{input_idx}",
                task.route,
                candidate_model.SerializeToString(),
                "generated",
                f"replace {output_name}={node.op_type}(...) with {input_name}",
            )


def profile(raw: bytes, candidate_count: int) -> ProfileSummary:
    model = onnx.load_model_from_string(raw)
    return ProfileSummary(
        task_id=TASK_ID,
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        file_bytes=len(raw),
        op_counts=json.dumps(dict(Counter(node.op_type for node in model.graph.node).most_common()), ensure_ascii=False),
        candidate_count=candidate_count,
    )


def selected_row(task_id: int, base: BaseTask, accepted_eval: dict[str, object] | None, raw: bytes) -> dict[str, object]:
    if accepted_eval is None:
        return {
            "task_id": task_id,
            "source": base.source,
            "template_name": base.template_name,
            "route": base.route,
            "cost": base.cost,
            "local_points": base.points,
            "file_bytes": len(base.raw),
            "status": "baseline",
            "reason": "no task037 full-arc graph-surgery gain",
            "sha256": sha256(base.raw),
        }
    return {
        "task_id": task_id,
        "source": EXP_ID,
        "template_name": accepted_eval["template_name"],
        "route": base.route,
        "cost": accepted_eval["candidate_cost"],
        "local_points": accepted_eval["candidate_points"],
        "file_bytes": len(raw),
        "status": "improved",
        "reason": accepted_eval["reason"],
        "sha256": sha256(raw),
    }


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_strict_seed_tasks()
    task = base_tasks[TASK_ID]
    candidates = list(bypass_candidates(task))
    eval_rows: list[dict[str, object]] = []
    accepted: tuple[dict[str, object], bytes] | None = None
    for candidate in candidates:
        evaluation, raw = evaluate_candidate(utils, candidate, task, -1, EXP_DIR)
        row = asdict(evaluation)
        eval_rows.append(row)
        if raw is None or row["status"] != "improved":
            continue
        if accepted is None or float(row["candidate_points"]) > float(accepted[0]["candidate_points"]):
            accepted = (row, raw)

    final_raw = {task_id: base.raw for task_id, base in base_tasks.items()}
    if accepted is not None:
        final_raw[TASK_ID] = accepted[1]
    write_zip(OUTPUT_ZIP, final_raw)

    selected_rows = [
        selected_row(task_id, base, accepted[0] if task_id == TASK_ID and accepted is not None else None, final_raw[task_id])
        for task_id, base in sorted(base_tasks.items())
    ]
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)
    fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    delta = 0.0
    if accepted is not None:
        delta = float(accepted[0]["candidate_points"]) - task.points
    prof = profile(task.raw, len(candidates))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_submit_candidate" if accepted is not None else "no_gain",
        "task_id": TASK_ID,
        "baseline_exp": str(BASE_EXP.relative_to(ROOT)),
        "strict_seed_local_estimate": STRICT_SEED_SCORE,
        "baseline_task_cost": task.cost,
        "baseline_task_points": task.points,
        "profile": asdict(prof),
        "candidate_status_counts": dict(Counter(str(row["status"]) for row in eval_rows)),
        "candidate_count": len(candidates),
        "accepted": accepted[0] if accepted is not None else None,
        "local_estimate_delta": delta,
        "new_local_estimate": STRICT_SEED_SCORE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_recommended_for_single_task_delta_calibration" if accepted is not None else "no_submit: no cost gain",
        "leakage_risk": "low-to-medium: graph surgery keeps strict artifact base and full arc-gen gate; no teacher artifact copied.",
        "overfitting_risk": "medium: full arc-gen passes are required, but hidden generated cases still need Kaggle delta if improved.",
        "decision": "submit single-task delta if improved; otherwise continue with rule-specific low-cost lowering.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task037 strict artifactを全arc-gen gate付きでgraph surgeryし、b010で見つかったruleに近い既存実装を安全に削れるか確認する。

## 結果

- node count: {prof.node_count}
- initializer count: {prof.initializer_count}
- candidate count: {len(candidates)}
- status counts: {result["candidate_status_counts"]}
- accepted: {result["accepted"]}
- local estimate delta: {delta:.6f}

## 判断

改善があればsingle-task deltaとしてKaggle較正候補。改善がなければ、既存strict artifact surgeryではなくrule-specific loweringへ戻る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

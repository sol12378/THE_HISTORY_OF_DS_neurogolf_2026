from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    candidate_fields,
    evaluate_candidate,
    load_neurogolf_utils,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp068_seddik_style_strict_scalarization"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp066_task020_correctness_onnx_lowering"
AUDIT_CSV = ROOT / "experiments" / "exp067_public_notebook_utilization_audit" / "seddik_surgery_audit.csv"
BASE_SCORE = 6282.43995971857
OUTPUT_ZIP = EXP_DIR / "submission.zip"

SAFE_UNIFORM_OPS = {
    "Greater",
    "Less",
    "Equal",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Where",
    "Max",
    "Min",
    "And",
    "Or",
    "Not",
    "Clip",
    "LessOrEqual",
    "GreaterOrEqual",
    "Sum",
}


@dataclass(frozen=True)
class SurgeryDetail:
    task_id: int
    unused_removed: int
    duplicate_rewired: int
    uniform_scalarized: int
    params_saved_proxy: int
    status: str
    reason: str


def row_cost_points(row: dict[str, str]) -> tuple[int, float]:
    if row.get("cost", ""):
        cost = int(float(row["cost"]))
    elif row.get("candidate_cost", ""):
        cost = int(float(row["candidate_cost"]))
    else:
        raise KeyError(row)
    if row.get("local_points", ""):
        pts = float(row["local_points"])
    elif row.get("candidate_points", ""):
        pts = float(row["candidate_points"])
    else:
        pts = point(cost)
    return cost, pts


def load_base_tasks(exp_dir: pathlib.Path) -> dict[int, BaseTask]:
    manifest_path = exp_dir / "selected_manifest.csv"
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    out: dict[int, BaseTask] = {}
    with manifest_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            cost, pts = row_cost_points(row)
            out[task_id] = BaseTask(
                task_id=task_id,
                cost=cost,
                points=pts,
                source=row.get("source", exp_dir.name),
                template_name=row.get("template_name", ""),
                route=row.get("route", ""),
                raw=raws[task_id],
            )
    return out


def audit_target_tasks() -> list[int]:
    if not AUDIT_CSV.exists():
        return list(range(1, 401))
    rows = []
    with AUDIT_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            score = int(row["total_seddik_style_saveable"])
            if score > 0:
                rows.append((score, int(row["task_id"])))
    rows.sort(reverse=True)
    return [task_id for _, task_id in rows]


def initializer_key(init: onnx.TensorProto) -> tuple[str, tuple[int, ...], bytes]:
    arr = numpy_helper.to_array(init)
    return (arr.dtype.str, tuple(arr.shape), arr.tobytes())


def array_params(init: onnx.TensorProto) -> int:
    return int(numpy_helper.to_array(init).size)


def replace_initializer(model: onnx.ModelProto, old_name: str, new_init: onnx.TensorProto) -> None:
    graph = model.graph
    for idx, init in enumerate(graph.initializer):
        if init.name == old_name:
            graph.initializer.remove(init)
            graph.initializer.insert(idx, new_init)
            return
    graph.initializer.append(new_init)


def apply_seddik_surgery(raw: bytes, task_id: int) -> tuple[bytes | None, SurgeryDetail]:
    model = onnx.load_from_string(raw)
    graph = model.graph
    used = {inp for node in graph.node for inp in node.input if inp}
    consumers: dict[str, list[str]] = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node.op_type)

    unused_removed = 0
    params_saved_proxy = 0
    for init in list(graph.initializer):
        if init.name not in used:
            params_saved_proxy += array_params(init)
            graph.initializer.remove(init)
            unused_removed += 1

    canonical_by_key: dict[tuple[str, tuple[int, ...], bytes], str] = {}
    duplicate_rewired = 0
    for init in list(graph.initializer):
        key = initializer_key(init)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = init.name
            continue
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp == init.name:
                    node.input[i] = canonical
                    duplicate_rewired += 1
        params_saved_proxy += array_params(init)
        graph.initializer.remove(init)

    used = {inp for node in graph.node for inp in node.input if inp}
    consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node.op_type)

    uniform_scalarized = 0
    for init in list(graph.initializer):
        if init.name not in used:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size <= 1:
            continue
        if not np.all(arr == arr.flat[0]):
            continue
        ops = consumers.get(init.name, [])
        if not ops or not all(op in SAFE_UNIFORM_OPS for op in ops):
            continue
        scalar = np.asarray(arr.flat[0], dtype=arr.dtype)
        replace_initializer(model, init.name, numpy_helper.from_array(scalar, init.name))
        params_saved_proxy += int(arr.size - 1)
        uniform_scalarized += 1

    changed = unused_removed + duplicate_rewired + uniform_scalarized
    if changed == 0:
        return None, SurgeryDetail(task_id, unused_removed, duplicate_rewired, uniform_scalarized, params_saved_proxy, "skipped", "no change")
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        return None, SurgeryDetail(task_id, unused_removed, duplicate_rewired, uniform_scalarized, params_saved_proxy, "rejected", f"onnx check failed: {str(exc)[:160]}")
    return model.SerializeToString(), SurgeryDetail(task_id, unused_removed, duplicate_rewired, uniform_scalarized, params_saved_proxy, "generated", "ok")


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks(BASE_EXP)
    target_tasks = audit_target_tasks()
    eval_rows = []
    detail_rows: list[SurgeryDetail] = []
    final_raws = {task_id: task.raw for task_id, task in base_tasks.items()}
    improved_tasks: list[int] = []
    accepted_delta = 0.0

    for task_id in target_tasks:
        base = base_tasks[task_id]
        raw, detail = apply_seddik_surgery(base.raw, task_id)
        detail_rows.append(detail)
        candidate = Candidate(task_id, "seddik_style_scalarize_dedup_prune", base.route, raw, detail.status, detail.reason)
        eval_row, accepted_raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        eval_rows.append(eval_row)
        if accepted_raw is not None and eval_row.status == "improved":
            final_raws[task_id] = accepted_raw
            improved_tasks.append(task_id)
            accepted_delta += float(eval_row.candidate_points) - eval_row.baseline_points

    write_zip(OUTPUT_ZIP, final_raws)
    selected_rows: list[dict[str, Any]] = []
    eval_by_task = {row.task_id: row for row in eval_rows}
    for task_id in sorted(base_tasks):
        task = base_tasks[task_id]
        row = eval_by_task.get(task_id)
        if row is not None and row.status == "improved":
            raw = final_raws[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": f"{EXP_ID}_{row.template_name}",
                    "template_name": row.template_name,
                    "route": task.route,
                    "cost": row.candidate_cost,
                    "local_points": row.candidate_points,
                    "file_bytes": len(raw),
                    "status": "improved",
                    "reason": row.reason,
                    "sha256": sha256(raw),
                }
            )
        else:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": task.source,
                    "template_name": task.template_name,
                    "route": task.route,
                    "cost": task.cost,
                    "local_points": task.points,
                    "file_bytes": len(task.raw),
                    "status": "baseline",
                    "reason": "no accepted seddik-style gain",
                    "sha256": sha256(task.raw),
                }
            )

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows([asdict(row) for row in eval_rows])
    with (EXP_DIR / "surgery_detail.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SurgeryDetail.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in detail_rows])
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    new_score = BASE_SCORE + accepted_delta
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_submit_candidate" if improved_tasks else "no_gain",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "baseline_local_estimate": BASE_SCORE,
        "new_local_estimate": new_score,
        "delta": accepted_delta,
        "tasks_targeted": len(target_tasks),
        "tasks_improved": improved_tasks,
        "eval_status_counts": dict(Counter(row.status for row in eval_rows)),
        "validation_status_counts": dict(Counter(row.validation_status for row in eval_rows)),
        "proxy_params_saved_generated": sum(row.params_saved_proxy for row in detail_rows if row.status == "generated"),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_if_improved_and_user_policy_allows" if improved_tasks else "no_submit",
        "leakage_risk": "low: graph surgery only, no outputs or labels used.",
        "overfitting_risk": "low: candidates accepted only with full arc-gen validation.",
        "decision": "If improved, submit as another strict-safe micro-delta to calibrate local/LB. If no gain, use audit as guardrail and move back to rule compiler.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

Seddik notebookの外科的ONNX圧縮を、現在のsubmit-safe base `exp066` にfull-arc gated post-passとして適用する。

## 結果

- targeted tasks: {len(target_tasks)}
- improved tasks: {improved_tasks}
- local delta: {accepted_delta:.9f}
- new local estimate: {new_score:.9f}

## 手法

- unused initializer prune
- exact duplicate initializer dedup
- uniform initializer scalarization for broadcast-safe consumer ops

## Decision

改善があればmicro-deltaとして提出し、local/LB一致を確認する。改善がなければSeddik-style surgeryはpost-pass guardrailとして残し、rule compilerへ戻る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from collections import Counter

    main()

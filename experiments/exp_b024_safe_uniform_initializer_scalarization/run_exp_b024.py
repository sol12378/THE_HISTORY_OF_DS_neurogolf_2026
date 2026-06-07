from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter, defaultdict
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
    load_routes,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp_b024_safe_uniform_initializer_scalarization"
EXP_DIR = ROOT / "experiments" / EXP_ID
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
STRICT_SEED_SCORE = 6282.230228
TOP_K = 80
SAFE_OPS = {
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
class CompressCandidate:
    name: str
    shape: str
    dtype: str
    value: float
    saveable_params: int
    consumer_ops: str


def load_strict_seed_tasks() -> dict[int, BaseTask]:
    routes = load_routes()
    rows: dict[int, dict[str, str]] = {}
    with (STRICT_EXP / "rewrite_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(STRICT_EXP / "submission.zip") as zf:
        raws = {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}
    return {
        task_id: BaseTask(
            task_id=task_id,
            cost=int(float(row["new_cost"])),
            points=float(row["new_points"]),
            source=row["source"],
            template_name="strict_seed",
            route=routes.get(task_id, ""),
            raw=raws[task_id],
        )
        for task_id, row in rows.items()
    }


def top_task_ids(tasks: dict[int, BaseTask]) -> list[int]:
    return [t.task_id for t in sorted(tasks.values(), key=lambda x: (-x.cost, x.task_id))[:TOP_K]]


def find_uniform_safe_initializers(model: onnx.ModelProto) -> list[CompressCandidate]:
    consumers: dict[str, list[str]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node.op_type)
    out = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        size = int(arr.size)
        if size <= 1:
            continue
        flat = arr.reshape(-1)
        if not np.all(flat == flat[0]):
            continue
        ops = consumers.get(init.name, [])
        if not ops or not all(op in SAFE_OPS for op in ops):
            continue
        out.append(
            CompressCandidate(
                name=init.name,
                shape=json.dumps(list(arr.shape)),
                dtype=str(arr.dtype),
                value=float(flat[0]),
                saveable_params=size - 1,
                consumer_ops=json.dumps(ops, ensure_ascii=False),
            )
        )
    return out


def compress_uniform_initializers(raw: bytes) -> tuple[bytes | None, list[CompressCandidate], str]:
    model = onnx.load_model_from_string(raw)
    candidates = find_uniform_safe_initializers(model)
    if not candidates:
        return None, [], "no safe uniform initializers"
    cand_names = {c.name for c in candidates}
    new_inits = []
    for init in model.graph.initializer:
        if init.name not in cand_names:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init)
        scalar = np.asarray(arr.reshape(-1)[0], dtype=arr.dtype)
        new_inits.append(numpy_helper.from_array(scalar, init.name))
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_inits)
    return model.SerializeToString(), candidates, "generated"


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_strict_seed_tasks()
    eval_rows = []
    candidate_rows = []
    accepted: dict[int, tuple[Any, bytes]] = {}
    generated = 0
    for task_id in top_task_ids(base):
        task = base[task_id]
        raw, compress_candidates, reason = compress_uniform_initializers(task.raw)
        for c in compress_candidates:
            row = asdict(c)
            row["task_id"] = task_id
            candidate_rows.append(row)
        if raw is None:
            continue
        generated += 1
        candidate = Candidate(task_id, "safe_uniform_initializer_scalarization", task.route, raw, "generated", reason)
        evaluation, accepted_raw = evaluate_candidate(utils, candidate, task, -1, EXP_DIR)
        eval_rows.append(asdict(evaluation))
        if accepted_raw is not None and evaluation.status == "improved":
            accepted[task_id] = (evaluation, accepted_raw)
    final_raw = {task_id: task.raw for task_id, task in base.items()}
    for task_id, (_evaluation, raw) in accepted.items():
        final_raw[task_id] = raw
    if accepted:
        write_zip(OUTPUT_ZIP, final_raw)
    with (EXP_DIR / "candidate_initializers.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id"] + list(CompressCandidate.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)
    selected_rows = []
    for task_id, task in sorted(base.items()):
        if task_id in accepted:
            evaluation, raw = accepted[task_id]
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": EXP_ID,
                    "template_name": evaluation.template_name,
                    "route": task.route,
                    "cost": evaluation.candidate_cost,
                    "local_points": evaluation.candidate_points,
                    "file_bytes": len(raw),
                    "status": "improved",
                    "reason": evaluation.reason,
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
                    "reason": "strict seed",
                    "sha256": sha256(task.raw),
                }
            )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)
    delta = sum(float(ev.candidate_points) - base[task_id].points for task_id, (ev, _raw) in accepted.items())
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_submit_candidate" if accepted else "no_gain",
        "source_pattern_exp": "exp_b023_seddik_surgery_pattern_audit",
        "strict_exp": str(STRICT_EXP.relative_to(ROOT)),
        "top_k": TOP_K,
        "screened_task_count": len(top_task_ids(base)),
        "safe_initializer_rows": len(candidate_rows),
        "generated_candidate_count": generated,
        "candidate_status_counts": dict(Counter(row["status"] for row in eval_rows)),
        "improved_task_count": len(accepted),
        "improved_tasks": sorted(accepted),
        "local_estimate_delta": delta,
        "new_local_estimate": STRICT_SEED_SCORE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP) if accepted else {},
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_micro_delta_if_improved" if accepted else "no_submit: no cost gain",
        "leakage_risk": "low-to-medium: public pattern but applied to strict seed with full-arc gate.",
        "overfitting_risk": "medium: scalarization is semantic only under safe broadcast assumptions; full-arc validation required.",
        "decision": "submit if improved; otherwise audit duplicate initializers next.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

seddik notebook由来のuniform initializer scalarizationをstrict seed top{TOP_K} taskへ安全ゲート付きで試す。

## 結果

- safe initializer rows: {len(candidate_rows)}
- generated candidates: {generated}
- improved tasks: {sorted(accepted)}
- local delta: {delta:.6f}
- decision: {result["submission_decision"]}

## Risk

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

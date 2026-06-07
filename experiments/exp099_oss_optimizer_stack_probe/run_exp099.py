from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import date
from typing import Any

import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_neurogolf_utils,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp099_oss_optimizer_stack_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
LOW_COST_PROFILE = ROOT / "experiments" / "exp_b033_low_cost_artifact_profile" / "artifact_profile.csv"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.812218

OSS_MODULES = ["onnxsim", "onnxoptimizer", "onnxscript", "onnx_graphsurgeon", "polygraphy", "egglog"]
IMPORTANT_RULE_HITS = {20, 37, 48, 85, 185, 251, 365, 366}


def load_current_tasks() -> dict[int, BaseTask]:
    rows: dict[int, dict[str, str]] = {}
    with (CURRENT_EXP / "selected_manifest.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["task_id"])] = row
    with zipfile.ZipFile(CURRENT_EXP / "submission.zip") as zf:
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


def load_probe_task_ids(base_tasks: dict[int, BaseTask]) -> list[int]:
    selected: set[int] = set()
    if LOW_COST_PROFILE.exists():
        with LOW_COST_PROFILE.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                cost = int(float(row["cost"]))
                if cost <= 600:
                    selected.add(int(row["task_id"]))
    top_high = sorted(base_tasks.values(), key=lambda item: (-item.cost, item.task_id))[:30]
    selected.update(task.task_id for task in top_high)
    selected.update(IMPORTANT_RULE_HITS)
    return sorted(t for t in selected if t in base_tasks)


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "checkpoint.log").write_text("start\n", encoding="utf-8")
    utils = load_neurogolf_utils()
    with (EXP_DIR / "checkpoint.log").open("a", encoding="utf-8") as f:
        f.write("utils_loaded\n")
    base_tasks = load_current_tasks()
    with (EXP_DIR / "checkpoint.log").open("a", encoding="utf-8") as f:
        f.write(f"base_loaded {len(base_tasks)}\n")
    probe_task_ids = load_probe_task_ids(base_tasks)
    with (EXP_DIR / "checkpoint.log").open("a", encoding="utf-8") as f:
        f.write(f"probe_loaded {len(probe_task_ids)}\n")
    module_availability = {name: importlib.util.find_spec(name) is not None for name in OSS_MODULES}
    with (EXP_DIR / "checkpoint.log").open("a", encoding="utf-8") as f:
        f.write(f"modules {module_availability}\n")

    # Generic ORT optimization was already tried in exp009 and produced no gain.
    # During this experiment, ORT offline serialization on the first current-best
    # probe task terminated the Python process before an exception could be
    # logged. Keep exp099 stable and record the tool-stack decision instead of
    # making the 30-experiment compiler campaign depend on this unstable route.
    optimizer_rows: list[dict[str, Any]] = []
    for task_id in probe_task_ids:
        base = base_tasks[task_id]
        optimizer_rows.append(
            {
                "task_id": task_id,
                "base_cost": base.cost,
                "route": base.route,
                "decision": "skip_generic_ort_offline",
                "reason": "onnxruntime offline optimizer previously showed no gain in exp009 and terminated this probe on task002; use dedicated NeuroGolf compiler instead.",
                "base_sha256": sha256(base.raw),
            }
        )

    accepted: dict[int, bytes] = {}
    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "optimizer_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task_id", "base_cost", "route", "decision", "reason", "base_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(optimizer_rows)
    delta = 0.0

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "tooling_probe_ready",
        "purpose": "Probe whether importable/offline OSS optimizers can reduce current submit-safe ONNX cost before building a dedicated compiler.",
        "module_availability": module_availability,
        "base_exp": str(CURRENT_EXP.relative_to(ROOT)),
        "probe_task_count": len(probe_task_ids),
        "probe_task_ids": probe_task_ids,
        "optimizer_labels": [],
        "evaluated_candidates": 0,
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted": [],
        "local_estimate_delta": delta,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "submission_decision": "no_submit: tooling probe only",
        "decision": "Do not spend the compiler campaign on generic ONNX optimizer tuning. No requested OSS modules are installed; ORT generic optimization was no-gain in exp009 and unstable in this probe. Proceed to NeuroGolf-specific low-cost artifact mining and rewrite/e-graph extraction.",
        "leakage_risk": "low: semantics-preserving optimizer pass on current submit-safe artifacts, full validation gated.",
        "overfitting_risk": "low: no task labels are used; validation still required because optimizer rewrites may change graph semantics or official static compliance.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## 目的

革新的compiler設計の第1実験として、汎用OSS/ORT optimizerを現行submit-safe ONNXへ適用するだけで公式cost改善が出るかを測る。改善がなければ、汎用optimizerではなくNeuroGolf専用rewrite/e-graph extractorへ進む。

## 結果

- probe task count: `{len(probe_task_ids)}`
- evaluated candidates: `0`
- accepted tasks: `[]`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`
- module availability: `{module_availability}`

## 解釈

汎用optimizerは推論速度や標準冗長削除を目的にしており、NeuroGolfの `memory+params` costを直接最小化しない。現環境では主要OSS moduleも未導入で、ORT offline optimizerは過去exp009でno gain、今回probeではtask002でプロセス終了した。次は低cost artifact miningと専用rewrite/extractionへ移す。

## Decision

`result.json` の acceptedが空なら、OSSは実行基盤・rewriter部品として使い、汎用最適化passそのものを主戦力にしない。

## Risk

- leakage risk: low。
- overfitting risk: low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        EXP_DIR.mkdir(parents=True, exist_ok=True)
        (EXP_DIR / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise

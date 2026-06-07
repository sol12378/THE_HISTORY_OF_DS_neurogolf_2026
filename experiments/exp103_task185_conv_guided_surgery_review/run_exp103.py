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

import numpy as np
import onnx
from onnx import numpy_helper

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


EXP_ID = "exp103_task185_conv_guided_surgery_review"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
CAMPAIGN_PROGRESS = ROOT / "experiments" / "compiler_campaign_30" / "progress.md"
OUTPUT_ZIP = EXP_DIR / "submission.zip"
CURRENT_LOCAL_ESTIMATE = 6282.812218
TASK_ID = 185


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


def graph_inventory(raw: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(raw)
    produced = {out for node in model.graph.node for out in node.output}
    consumed = {inp for node in model.graph.node for inp in node.input}
    graph_outputs = {out.name for out in model.graph.output}
    unused_nodes = []
    for idx, node in enumerate(model.graph.node):
        if not any(out in consumed or out in graph_outputs for out in node.output):
            unused_nodes.append({"idx": idx, "op": node.op_type, "outputs": list(node.output)})
    init_hashes: dict[str, list[str]] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        key = f"{arr.dtype}:{arr.shape}:{sha256(arr.tobytes())}"
        init_hashes.setdefault(key, []).append(init.name)
    duplicate_inits = [names for names in init_hashes.values() if len(names) > 1]
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "file_bytes": len(raw),
        "op_counts": dict(sorted(__import__("collections").Counter(n.op_type for n in model.graph.node).items())),
        "unused_node_count": len(unused_nodes),
        "unused_nodes": unused_nodes[:20],
        "duplicate_initializer_groups": duplicate_inits[:20],
        "duplicate_initializer_group_count": len(duplicate_inits),
        "produced_value_count": len(produced),
        "consumed_value_count": len(consumed),
    }


def dedupe_initializers(raw: bytes) -> bytes:
    model = onnx.load_model_from_string(raw)
    seen: dict[str, str] = {}
    replace: dict[str, str] = {}
    kept = []
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        key = f"{arr.dtype}:{arr.shape}:{sha256(arr.tobytes())}"
        if key in seen:
            replace[init.name] = seen[key]
        else:
            seen[key] = init.name
            kept.append(init)
    if not replace:
        return raw
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in replace:
                node.input[i] = replace[inp]
    return model.SerializeToString()


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_current_tasks()
    base = base_tasks[TASK_ID]
    inventory = graph_inventory(base.raw)

    candidates = [
        Candidate(TASK_ID, "sanitize_current_task185", base.route, base.raw, "generated", "official sanitize/score only"),
        Candidate(TASK_ID, "dedupe_initializers_task185", base.route, dedupe_initializers(base.raw), "generated", "dedupe identical initializers then validate"),
    ]
    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval: CandidateEval | None = None
    for candidate in candidates:
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is not None and ev.status == "improved" and (best_eval is None or int(ev.candidate_cost) < int(best_eval.candidate_cost)):
            accepted[TASK_ID] = raw
            best_eval = ev

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(CandidateEval.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    delta = 0.0 if best_eval is None else float(best_eval.candidate_points) - float(best_eval.baseline_points)
    review = {
        "window": "compiler_campaign_30 experiments #1-#5",
        "experiments": ["exp099", "exp100", "exp101", "exp102", "exp103"],
        "local_delta_sum": delta,
        "lb_usefulness": "mixed_but_directionally_useful",
        "verdict": "No local/LB improvement yet, but the five experiments moved from generic optimizer hopes to a NeuroGolf-specific compiler grammar and ruled out simple archetype scans. This is useful only if the next block emits candidates from computed_slice_pad/one-Conv patterns rather than continuing catalogs.",
        "next_policy": "The next 5 experiments must include at least two score-producing candidate emissions, not only mining. Prioritize computed_slice_pad/tiny_dynamic_shape_index and task185/task087-style cost floor reductions.",
    }
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "campaign_index": 5,
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "inventory": inventory,
        "candidate_eval": [asdict(ev) for ev in evals],
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": delta,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE + delta,
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "five_experiment_review": review,
        "runtime_seconds": time.time() - started,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "decision": "task185 has 46 lattice-position patterns; static Conv proxy is not directly submit-safe. Simple sanitize/dedupe surgery is tested here; if no gain, move to computed position/index compiler rather than more task185 micro-surgery.",
        "leakage_risk": "low: graph surgery/diagnostic only.",
        "overfitting_risk": "low for surgery; medium for future task185 position compiler.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

compiler campaign #5。task185のone-Conv方針を、既存artifact surgeryと格子位置inventoryで現実評価する。5実験レビューも同時に行う。

## 結果

- accepted tasks: `{sorted(accepted)}`
- local delta: `{delta:.6f}`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE + delta:.6f}`
- task185 baseline cost: `{base.cost}`
- task185 node count: `{inventory['node_count']}`
- duplicate initializer groups: `{inventory['duplicate_initializer_group_count']}`

## 5実験レビュー

{review['verdict']}

## Decision

task185は46格子位置patternがあり、static Conv proxyをそのまま提出候補にはできない。simple surgeryで改善がない場合、次blockは `computed_slice_pad` / `tiny_dynamic_shape_index` compilerとして位置推定を扱う。

## Risk

- leakage risk: low。
- overfitting risk: low〜medium。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    if CAMPAIGN_PROGRESS.exists():
        with CAMPAIGN_PROGRESS.open("a", encoding="utf-8") as f:
            f.write(f"\n| 5 | exp103_task185_conv_guided_surgery_review | task185 one-Conv方針をsurgery/inventoryで評価し5実験レビュー | {delta:.6f} | {CURRENT_LOCAL_ESTIMATE + delta:.6f} | accepted {sorted(accepted)}; nextはcomputed position/index compiler |\n")
            f.write("\n## Review After #5\n\n")
            f.write(f"- verdict: {review['lb_usefulness']}\n")
            f.write(f"- local delta #1-#5: `{delta:.6f}` (all campaign deltas currently 0 unless exp103 accepted)\n")
            f.write(f"- analysis: {review['verdict']}\n")
            f.write(f"- next policy: {review['next_policy']}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

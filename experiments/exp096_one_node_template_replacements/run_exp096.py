from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict
from datetime import date
from typing import Any

import numpy as np
from onnx import helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import (  # noqa: E402
    Candidate,
    CandidateEval,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    make_model,
    point,
    sha256,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp096_one_node_template_replacements"
EXP_DIR = ROOT / "experiments" / EXP_ID
SCAN_HITS = ROOT / "experiments" / "exp095_one_node_template_rule_scan" / "one_node_template_hits.csv"
STRICT_SEED_SCORE = 6282.812218
OUTPUT_ZIP = EXP_DIR / "submission.zip"


def init_i(name: str, value: np.ndarray) -> Any:
    return numpy_helper.from_array(value.astype(np.int64), name)


def raw_flip(axis: int, producer: str) -> bytes:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])]
    inits = [
        init_i("starts", np.asarray([-1])),
        init_i("ends", np.asarray([-(1 << 31)])),
        init_i("axes", np.asarray([axis])),
        init_i("steps", np.asarray([-1])),
    ]
    return make_model(nodes, inits, producer, opset_version=11)


def raw_rot180(producer: str) -> bytes:
    nodes = [helper.make_node("Slice", ["input", "starts", "ends", "axes", "steps"], ["output"])]
    inits = [
        init_i("starts", np.asarray([-1, -1])),
        init_i("ends", np.asarray([-(1 << 31), -(1 << 31)])),
        init_i("axes", np.asarray([2, 3])),
        init_i("steps", np.asarray([-1, -1])),
    ]
    return make_model(nodes, inits, producer, opset_version=11)


def raw_channel_gather(mapping: dict[int, int], producer: str) -> bytes:
    # For one-hot input, output channel c should gather source channel src where
    # mapping[src] == c. If multiple source colors map to one output color, a
    # single Gather is not sufficient; skip those cases.
    inv: dict[int, int] = {}
    for src, dst in mapping.items():
        if dst in inv and inv[dst] != src:
            raise ValueError("non-injective mapping cannot be channel Gather")
        inv[dst] = src
    idx = np.asarray([inv.get(c, c) for c in range(10)], dtype=np.int64)
    nodes = [helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)]
    return make_model(nodes, [init_i("idx", idx)], producer, opset_version=11)


def build_candidate(task_id: int, variant: str, mapping_s: str, route: str) -> Candidate | None:
    producer = f"{EXP_ID}_task{task_id:03d}_{variant}"
    if variant == "flip_lr":
        raw = raw_flip(3, producer)
    elif variant == "flip_ud":
        raw = raw_flip(2, producer)
    elif variant == "rot180":
        raw = raw_rot180(producer)
    elif variant.endswith("_channel_recolor"):
        mapping = {int(k): int(v) for k, v in json.loads(mapping_s).items()}
        base_variant = variant.replace("_channel_recolor", "")
        if base_variant != "identity":
            return None
        try:
            raw = raw_channel_gather(mapping, producer)
        except ValueError:
            return None
    else:
        return None
    return Candidate(task_id, variant, route, raw, "generated", "one-node official template replacement")


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks()
    hit_rows: list[dict[str, str]] = []
    with SCAN_HITS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # Ignore already near-perfect zero/10-cost baselines and expensive rot90 proxy.
            if int(float(row["baseline_cost"])) <= 20:
                continue
            if row["variant"].startswith("rot90"):
                continue
            if row["variant"].endswith("_channel_recolor") and not row["variant"].startswith("identity"):
                continue
            hit_rows.append(row)

    evals: list[CandidateEval] = []
    accepted: dict[int, bytes] = {}
    best_eval_by_task: dict[int, CandidateEval] = {}
    for row in hit_rows:
        task_id = int(row["task_id"])
        base = base_tasks[task_id]
        candidate = build_candidate(task_id, row["variant"], row["mapping"], base.route)
        if candidate is None:
            continue
        ev, raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        evals.append(ev)
        if raw is None or ev.status != "improved":
            continue
        current = best_eval_by_task.get(task_id)
        if current is None or int(ev.candidate_cost) < int(current.candidate_cost):
            accepted[task_id] = raw
            best_eval_by_task[task_id] = ev

    final_raw = {task_id: task.raw for task_id, task in base_tasks.items()}
    final_raw.update(accepted)
    write_zip(OUTPUT_ZIP, final_raw)

    fields = list(CandidateEval.__dataclass_fields__.keys())
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([asdict(ev) for ev in evals])

    selected_rows: list[dict[str, Any]] = []
    accepted_delta = 0.0
    for task_id in sorted(base_tasks):
        base = base_tasks[task_id]
        if task_id in best_eval_by_task:
            ev = best_eval_by_task[task_id]
            accepted_delta += float(ev.candidate_points) - float(ev.baseline_points)
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": EXP_ID,
                    "template_name": ev.template_name,
                    "route": base.route,
                    "cost": ev.candidate_cost,
                    "local_points": ev.candidate_points,
                    "file_bytes": len(accepted[task_id]),
                    "status": "improved",
                    "reason": ev.reason,
                    "sha256": sha256(accepted[task_id]),
                }
            )
        else:
            selected_rows.append(
                {
                    "task_id": task_id,
                    "source": base.source,
                    "template_name": base.template_name,
                    "route": base.route,
                    "cost": base.cost,
                    "local_points": base.points,
                    "file_bytes": len(base.raw),
                    "status": "baseline",
                    "reason": "no one-node replacement accepted",
                    "sha256": sha256(base.raw),
                }
            )
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fields2 = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields2)
        writer.writeheader()
        writer.writerows(selected_rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "improved_bundle_candidate" if accepted else "no_gain",
        "input_hits": len(hit_rows),
        "evaluated_candidates": len(evals),
        "accepted_tasks": sorted(accepted),
        "accepted_count": len(accepted),
        "accepted_delta": accepted_delta,
        "base_local_estimate": STRICT_SEED_SCORE,
        "new_local_estimate": STRICT_SEED_SCORE + accepted_delta,
        "accepted": [asdict(best_eval_by_task[t]) for t in sorted(best_eval_by_task)],
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "local_estimate_delta": accepted_delta,
        "submission_decision": "submit_candidate_after_review" if accepted else "no_submit",
        "leakage_risk": "low: deterministic one-node transforms validated on all arc-gen.",
        "overfitting_risk": "low for pure geometry; color Gather only accepted if full validation passes.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp095` のfull-hit one-node templateを公式ONNX replacementとして生成し、現在のbest bundleへ差し替え可能か評価する。

## 結果

- evaluated candidates: `{len(evals)}`
- accepted tasks: `{sorted(accepted)}`
- accepted delta: `{accepted_delta:.6f}`
- new local estimate: `{STRICT_SEED_SCORE + accepted_delta:.6f}`

## Decision

改善がある場合は、full validation済みmicro-deltaとしてsubmit候補にする。rot90は2ノード中間が重いので今回除外。

## Risk

- leakage risk: low。
- overfitting risk: low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

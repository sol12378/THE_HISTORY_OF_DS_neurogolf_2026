from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    Candidate,
    build_conv_color_map_candidate,
    build_global_transform_color_map_candidate,
    build_identity_candidate,
    candidate_fields,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp030_same_shape_transform_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
QUEUE_PATH = ROOT / "experiments" / "exp026_synthesis_orchestrator_core" / "synthesis_queue.csv"
ARC_GEN_SAMPLE = 20
TARGET_FAMILIES = {"same_shape_global_transform", "signature_lookup_current", "color_map"}


def load_target_tasks(base: dict[int, BaseTask], limit: int = 140) -> list[int]:
    tasks: list[int] = []
    with QUEUE_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            same_shape_hint = "same_shape" in row["synthesis_family"] or row["synthesis_family"] in TARGET_FAMILIES
            if same_shape_hint and task_id in base:
                tasks.append(task_id)
            if len(tasks) >= limit:
                break
    return tasks


def candidates_for(task_id: int, examples: list[dict], route: str) -> list[Candidate]:
    return [
        build_global_transform_color_map_candidate(task_id, examples, route),
        build_conv_color_map_candidate(task_id, examples, route),
        build_identity_candidate(task_id, examples, route),
    ]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)
    target_ids = load_target_tasks(base)

    eval_rows = []
    selected: dict[int, bytes] = {}
    selected_eval = {}
    generated = 0
    skipped = 0

    for task_id in target_ids:
        task = load_task(task_id)
        train_examples = examples_for(task, 0)[: len(task["train"])]
        for candidate in candidates_for(task_id, train_examples, base[task_id].route):
            if candidate.raw is None:
                skipped += 1
            else:
                generated += 1
            result, raw = evaluate_candidate(utils, candidate, base[task_id], ARC_GEN_SAMPLE, EXP_DIR)
            eval_rows.append(asdict(result))
            if raw is not None:
                old = selected_eval.get(task_id)
                if old is None or float(result.candidate_points) > float(old.candidate_points):
                    selected[task_id] = raw
                    selected_eval[task_id] = result

    bundle_raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            bundle_raws[task_id] = zf.read(name)
    bundle_raws.update(selected)

    sanity = {}
    if selected:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)

    selected_rows = [asdict(v) for _, v in sorted(selected_eval.items())]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    base_local = sum(task.points for task in base.values())
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in selected_rows)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "improved" if selected else "no_gain",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "target_task_count": len(target_ids),
        "candidate_rows": len(eval_rows),
        "generated_candidate_count": generated,
        "skipped_candidate_count": skipped,
        "improved_task_count": len(selected_rows),
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "selected_tasks": [row["task_id"] for row in selected_rows],
        "top_selected": sorted(selected_rows, key=lambda row: float(row["candidate_points"]) - float(row["baseline_points"]), reverse=True)[:10],
        "zip_sanity": sanity,
        "leakage_risk": "low-medium: train-fitted global transforms are interpretable, but base includes high-risk lookup artifacts.",
        "overfitting_risk": "medium: sample20 validation is not private-like family holdout.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## Hypothesis

same-shape系とlookup由来taskの一部は、trainだけから推定できる軽量な `Gather`/`Transpose` global transform、1x1 `Conv` color map、またはidentityでexp023 artifactより安く置換できる。

## Result

- base: `{result["base_exp"]}`
- target tasks: `{result["target_task_count"]}`
- candidate rows: `{result["candidate_rows"]}`
- generated candidates: `{result["generated_candidate_count"]}`
- improved tasks: `{result["improved_task_count"]}`
- local estimate: `{result["local_estimate"]:.6f}`
- delta: `{result["local_estimate_delta"]:.6f}`
- gap to 6500: `{result["gap_to_6500"]:.6f}`
- submission decision: `{result["submission_decision"]}`

## Interpretation

full-grid loweringやlarge `ScatterND` は使わず、cost guardrail上で軽い候補だけを試した。
改善が出ても6500未満なら提出しない。6500以上ならfull validationとrisk review後に提出候補にする。

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

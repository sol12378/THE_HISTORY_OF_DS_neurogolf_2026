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
    Candidate,
    BaseTask,
    build_common_crop_candidate,
    build_common_crop_color_map_candidate,
    candidate_fields,
    evaluate_candidate,
    examples_for,
    load_base_tasks,
    load_neurogolf_utils,
    load_task,
    point,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp029_crop_object_synthesizer"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
QUEUE_PATH = ROOT / "experiments" / "exp026_synthesis_orchestrator_core" / "synthesis_queue.csv"
ARC_GEN_SAMPLE = 20


def load_queue_crop_tasks(base: dict[int, BaseTask], limit: int = 120) -> list[int]:
    tasks: list[int] = []
    with QUEUE_PATH.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            if row["synthesis_family"] == "crop_resize" and task_id in base:
                tasks.append(task_id)
            if len(tasks) >= limit:
                break
    return tasks


def candidate_set(task_id: int, examples: list[dict], route: str) -> list[Candidate]:
    return [
        build_common_crop_candidate(task_id, examples, route),
        build_common_crop_color_map_candidate(task_id, examples, route),
    ]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks(BASE_EXP)
    target_ids = load_queue_crop_tasks(base)

    eval_rows = []
    selected: dict[int, bytes] = {}
    selected_eval = {}
    generated = 0
    skipped = 0

    for task_id in target_ids:
        task = load_task(task_id)
        train_examples = examples_for(task, 0)[: len(task["train"])]
        for candidate in candidate_set(task_id, train_examples, base[task_id].route):
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

    if selected:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")
    else:
        sanity = {}

    eval_path = EXP_DIR / "candidate_eval.csv"
    with eval_path.open("w", encoding="utf-8", newline="") as f:
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
        "leakage_risk": "medium: train-fitted crop rules are validated on test and arc-gen sample20; base still includes high-risk lookup artifacts.",
        "overfitting_risk": "medium: crop rules may be sample-specific; no private-like family holdout yet.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## Hypothesis

crop/resize系の一部は、trainだけから推定できる定数 `Slice` または定数 `Slice` + 1x1 `Conv` color mapで、exp023の高cost artifactより安く置換できる。

## Result

- base: `{result["base_exp"]}`
- target crop/resize tasks: `{result["target_task_count"]}`
- generated candidates: `{result["generated_candidate_count"]}`
- improved tasks: `{result["improved_task_count"]}`
- local estimate: `{result["local_estimate"]:.6f}`
- delta: `{result["local_estimate_delta"]:.6f}`
- gap to 6500: `{result["gap_to_6500"]:.6f}`
- submission decision: `{result["submission_decision"]}`

## Interpretation

失敗済みのfull-grid `Tile`、large `ScatterND`、signature lookupは使わず、cost guardrail上で軽い候補だけを検査した。
改善候補が出た場合も、baseがexp023由来のlocal upper boundを含むため、submit前にはfull validationとrisk reviewが必要。

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}

## Next

6500未満なら、crop候補の次は `same_shape_global_transform` または `line_grid_fill` の小さい `Gather`/`Conv` 系に限定して探索する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

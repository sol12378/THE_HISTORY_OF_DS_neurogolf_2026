from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    candidate_fields,
    load_neurogolf_utils,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp037_fullarc_filter_exp036"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp035_greedy_logic_surgery_composition"
SOURCE_EXP = ROOT / "experiments" / "exp036_noop_bypass_and_prune"


def load_zip_raws(path: pathlib.Path) -> dict[int, bytes]:
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            raws[task_id] = zf.read(name)
    return raws


def read_selected_manifest(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    source_result = json.loads((SOURCE_EXP / "result.json").read_text(encoding="utf-8"))
    base_result = json.loads((BASE_EXP / "result.json").read_text(encoding="utf-8"))
    base_raws = load_zip_raws(BASE_EXP / "submission.zip")
    source_raws = load_zip_raws(SOURCE_EXP / "submission.zip")
    source_rows = read_selected_manifest(SOURCE_EXP / "selected_manifest.csv")

    selected_rows = []
    rejected_rows = []
    validation_rows = []
    bundle_raws = dict(base_raws)
    for row in source_rows:
        task_id = int(row["task_id"])
        raw = source_raws[task_id]
        ok, reason, passed, failed = validate_examples(utils, raw, task_id, -1)
        validation_row = {
            "task_id": task_id,
            "full_arc_ok": ok,
            "passed": passed,
            "failed": failed,
            "reason": reason,
        }
        validation_rows.append(validation_row)
        if ok:
            kept = dict(row)
            kept["validation_status"] = f"full_arc_{passed}_pass_{failed}_fail"
            kept["reason"] = "kept after full arc-gen validation"
            selected_rows.append(kept)
            bundle_raws[task_id] = raw
        else:
            rejected = dict(row)
            rejected["validation_status"] = f"full_arc_{passed}_pass_{failed}_fail"
            rejected["status"] = "rejected_full_arc"
            rejected["reason"] = reason
            rejected_rows.append(rejected)

    if selected_rows:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")
    else:
        sanity = {}

    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    with (EXP_DIR / "rejected_fullarc.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(rejected_rows)

    with (EXP_DIR / "fullarc_validation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "full_arc_ok", "passed", "failed", "reason"])
        writer.writeheader()
        writer.writerows(validation_rows)

    base_local = float(base_result["local_estimate"])
    delta = sum(float(row["candidate_points"]) - float(row["baseline_points"]) for row in selected_rows)
    local_estimate = base_local + delta
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "improved" if selected_rows else "no_gain",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "source_exp": str(SOURCE_EXP.relative_to(ROOT)),
        "base_local_estimate": base_local,
        "source_local_estimate": source_result["local_estimate"],
        "source_selected_task_count": len(source_rows),
        "full_arc_kept_task_count": len(selected_rows),
        "full_arc_rejected_task_count": len(rejected_rows),
        "kept_tasks": [int(row["task_id"]) for row in selected_rows],
        "rejected_tasks": [int(row["task_id"]) for row in rejected_rows],
        "local_estimate_delta": delta,
        "local_estimate": local_estimate,
        "gap_to_6500": 6500.0 - local_estimate,
        "submission_decision": "submit_threshold_reached" if local_estimate >= 6500.0 else "no_submit: below 6500 threshold",
        "top_selected": sorted(selected_rows, key=lambda row: float(row["candidate_points"]) - float(row["baseline_points"]), reverse=True)[:20],
        "rejected_full_arc": rejected_rows,
        "zip_sanity": sanity,
        "leakage_risk": "medium: selected edits pass all available arc-gen validation, but base still contains high-risk lookup artifacts.",
        "overfitting_risk": "medium: full arc-gen filter removes observed sample20-only failures, but private distribution risk remains.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## Hypothesis

exp036の広いno-op bypass探索はsample20では有効だが、一部はarc-gen後半に過学習している可能性がある。全arc-genを通過した改善だけに絞れば、より堅いlocal estimateを更新できる。

## Result

- base: `{result["base_exp"]}`
- source: `{result["source_exp"]}`
- source selected tasks: `{result["source_selected_task_count"]}`
- kept after full arc-gen: `{result["full_arc_kept_task_count"]}`
- rejected after full arc-gen: `{result["full_arc_rejected_task_count"]}`
- local estimate: `{result["local_estimate"]:.6f}`
- delta vs base: `{result["local_estimate_delta"]:.6f}`
- gap to 6500: `{result["gap_to_6500"]:.6f}`
- submission decision: `{result["submission_decision"]}`

## Risks

- leakage risk: {result["leakage_risk"]}
- overfitting risk: {result["overfitting_risk"]}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import pathlib
import sys
import zipfile
from dataclasses import asdict
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "exp036_noop_bypass_and_prune"))

import run_exp036 as e36  # noqa: E402
from experiments.phase1_rewrite_utils import (  # noqa: E402
    BaseTask,
    candidate_fields,
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp253_task153_bypass_submit_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_BUNDLE = ROOT / "experiments" / "exp234_task300_max_color_submit_probe" / "submission.zip"
TASK_ID = 153
TARGET_TEMPLATE = "greedy1_Reshape_node7_to_input0"


def load_zip_raws(path: pathlib.Path) -> dict[int, bytes]:
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            raws[int(pathlib.Path(name).stem.replace("task", ""))] = zf.read(name)
    return raws


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks()
    raws = load_zip_raws(BASE_BUNDLE)
    old = base_tasks[TASK_ID]
    base = BaseTask(TASK_ID, old.cost, old.points, old.source, old.template_name, old.route, raws[TASK_ID])

    selected_eval = None
    selected_raw = None
    all_rows = []
    for candidate in e36.bypass_candidates(TASK_ID, raws[TASK_ID], base.route, 1):
        if candidate.template_name != TARGET_TEMPLATE:
            continue
        evaluation, accepted_raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        all_rows.append(asdict(evaluation))
        if accepted_raw is not None:
            ok, reason, passed, failed = validate_examples(utils, accepted_raw, TASK_ID, -1)
            row = asdict(evaluation)
            row["full_arc_ok"] = ok
            row["full_arc_passed"] = passed
            row["full_arc_failed"] = failed
            row["full_arc_reason"] = reason
            selected_eval = row
            if ok:
                selected_raw = accepted_raw
        break

    if not all_rows:
        raise RuntimeError(f"target candidate not generated: {TARGET_TEMPLATE}")

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = candidate_fields()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in all_rows])

    sanity = {}
    expected_public_lb = None
    if selected_raw is not None and selected_eval is not None:
        bundle = dict(raws)
        bundle[TASK_ID] = selected_raw
        write_zip(EXP_DIR / "submission.zip", bundle)
        sanity = zip_sanity(EXP_DIR / "submission.zip")
        expected_public_lb = 6006.32 + (float(selected_eval["candidate_points"]) - float(selected_eval["baseline_points"]))

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "base_bundle": str(BASE_BUNDLE.relative_to(ROOT)),
        "task_id": TASK_ID,
        "target_template": TARGET_TEMPLATE,
        "evaluation": selected_eval or all_rows[0],
        "improved_and_fullarc": selected_raw is not None,
        "zip_sanity": sanity,
        "expected_public_lb_if_calibrated": expected_public_lb,
        "decision": "Submit if improved_and_fullarc and zip_sanity names_ok.",
        "submission_decision": "submit_ready" if selected_raw is not None and sanity.get("names_ok") else "no_submit",
        "leakage_risk": "low-medium: semantics-preserving full-arc-gated graph surgery on current best bundle.",
        "overfitting_risk": "medium-low: full available validation; private robustness still needs LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp252で見つかったtask153 `Reshape_node7_to_input0` bypassを、current public best exp234 bundle上で再生成し、提出zipを作る。

## 結果

- improved_and_fullarc: `{result["improved_and_fullarc"]}`
- evaluation: `{result["evaluation"]}`
- zip_sanity: `{result["zip_sanity"]}`
- expected_public_lb_if_calibrated: `{result["expected_public_lb_if_calibrated"]}`

## 判断

`submit_ready`ならKaggleへ提出してmicro deltaを較正する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

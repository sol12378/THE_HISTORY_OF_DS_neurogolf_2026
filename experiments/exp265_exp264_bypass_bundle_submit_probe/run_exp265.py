from __future__ import annotations

import csv
import hashlib
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
    point,
    score_model,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp265_exp264_bypass_bundle_submit_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_BUNDLE = ROOT / "experiments" / "exp263_exp262_partial_bypass_bundle_submit_probe" / "submission.zip"
SOURCE_MANIFEST = ROOT / "experiments" / "exp264_current_rank321_400_fullarc_bypass_sweep" / "selected_manifest.csv"
EXPECTED_BASE_LB = 6008.30


def load_zip_raws(path: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def load_manifest(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def score_current_base(utils, raws: dict[int, bytes], proxy_base: dict[int, BaseTask], task_id: int) -> BaseTask:
    old = proxy_base[task_id]
    memory, params, reason = score_model(utils, raws[task_id], task_id, f"current_task{task_id:03d}", EXP_DIR)
    if memory is None or params is None:
        raise RuntimeError(f"score failed task{task_id:03d}: {reason}")
    cost = int(memory + params)
    return BaseTask(task_id, cost, point(cost), old.source, old.template_name, old.route, raws[task_id])


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    raws = load_zip_raws(BASE_BUNDLE)
    proxy_base = load_base_tasks()
    manifest_rows = load_manifest(SOURCE_MANIFEST)

    selected_rows = []
    audit_rows = []
    failures = []
    bundle_raws = dict(raws)

    for source_row in manifest_rows:
        task_id = int(source_row["task_id"])
        template_name = source_row["template_name"]
        try:
            base = score_current_base(utils, raws, proxy_base, task_id)
            match = None
            for candidate in e36.bypass_candidates(task_id, base.raw, base.route, 1):
                if candidate.template_name == template_name:
                    match = candidate
                    break
            if match is None:
                raise RuntimeError(f"candidate not found: {template_name}")
            evaluation, accepted_raw = evaluate_candidate(utils, match, base, 20, EXP_DIR)
            if accepted_raw is None:
                raise RuntimeError(f"candidate rejected on replay: {evaluation.reason}")
            ok, reason, passed, failed = validate_examples(utils, accepted_raw, task_id, -1)
            audit_rows.append(
                {
                    "task_id": task_id,
                    "template_name": template_name,
                    "full_arc_ok": ok,
                    "passed": passed,
                    "failed": failed,
                    "reason": reason,
                    "sha256": hashlib.sha256(accepted_raw).hexdigest(),
                }
            )
            if not ok:
                raise RuntimeError(f"full-arc failed: {reason}")
            selected_rows.append(asdict(evaluation))
            bundle_raws[task_id] = accepted_raw
        except Exception as exc:
            failures.append({"task_id": task_id, "template_name": template_name, "reason": f"{type(exc).__name__}: {exc}"})

    fields = candidate_fields()
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in selected_rows])

    with (EXP_DIR / "fullarc_replay_audit.csv").open("w", encoding="utf-8", newline="") as f:
        audit_fields = ["task_id", "template_name", "full_arc_ok", "passed", "failed", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    if selected_rows:
        write_zip(EXP_DIR / "submission.zip", bundle_raws)
        sanity = zip_sanity(EXP_DIR / "submission.zip")
    else:
        sanity = {}

    delta = sum(float(r["candidate_points"]) - float(r["baseline_points"]) for r in selected_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "base_bundle": str(BASE_BUNDLE.relative_to(ROOT)),
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "selected_count": len(selected_rows),
        "failed_count": len(failures),
        "failures": failures,
        "selected_tasks": [int(r["task_id"]) for r in selected_rows],
        "selected_rows": selected_rows,
        "local_delta": delta,
        "expected_public_lb_if_calibrated": EXPECTED_BASE_LB + delta,
        "zip_sanity": sanity,
        "submission_decision": "submit" if selected_rows and not failures else "review_before_submit",
        "leakage_risk": "low-medium: replayed exp264 full-arc gated graph surgery on current best bundle.",
        "overfitting_risk": "medium-low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp264でlocal estimateが更新された8件を、exp263 current bestに対して再生成し、full-arc replay gate後にbundle化する。

## 結果

- selected_count: `{len(selected_rows)}`
- failed_count: `{len(failures)}`
- selected_tasks: `{[int(r["task_id"]) for r in selected_rows]}`
- local_delta: `{delta}`
- expected_public_lb_if_calibrated: `{EXPECTED_BASE_LB + delta}`

## 判断

local estimate更新があり、replayで失敗がなければ提出対象。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

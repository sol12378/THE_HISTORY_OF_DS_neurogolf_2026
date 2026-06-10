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
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    score_model,
    validate_examples,
)


EXP_ID = "exp261_exp260_bypass_bundle_submit_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_BUNDLE = ROOT / "experiments" / "exp259_exp258_bypass_bundle_submit_probe" / "submission.zip"
MANIFEST = ROOT / "experiments" / "exp260_current_rank141_220_fullarc_bypass_sweep" / "selected_manifest.csv"
BASE_PUBLIC_LB = 6006.94


def load_zip_raws(path: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def load_selected_templates() -> dict[int, str]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return {int(row["task_id"]): row["template_name"] for row in csv.DictReader(f)}


def score_base_task(utils, raws: dict[int, bytes], proxy_base: dict[int, BaseTask], task_id: int) -> BaseTask:
    old = proxy_base[task_id]
    memory, params, reason = score_model(utils, raws[task_id], task_id, f"exp261_base_task{task_id:03d}", EXP_DIR)
    if memory is None or params is None:
        raise RuntimeError(f"score failed task{task_id:03d}: {reason}")
    cost = int(memory + params)
    return BaseTask(task_id, cost, point(cost), old.source, old.template_name, old.route, raws[task_id])


def zip_sanity(path: pathlib.Path) -> dict[str, object]:
    data = path.read_bytes()
    with zipfile.ZipFile(path) as zf:
        names = sorted(zf.namelist())
    return {
        "count": len(names),
        "names_ok": names == [f"task{i:03d}.onnx" for i in range(1, 401)],
        "first": names[:3],
        "last": names[-3:],
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    raws = load_zip_raws(BASE_BUNDLE)
    proxy_base = load_base_tasks()
    selected_templates = load_selected_templates()

    selected_rows = []
    selected_raws: dict[int, bytes] = {}
    failures = []
    for task_id, template_name in selected_templates.items():
        task = score_base_task(utils, raws, proxy_base, task_id)
        accepted = None
        accepted_eval = None
        for candidate in e36.bypass_candidates(task_id, task.raw, task.route, 1):
            if candidate.template_name != template_name:
                continue
            evaluation, accepted_raw = evaluate_candidate(utils, candidate, task, -1, EXP_DIR)
            if accepted_raw is None:
                failures.append({"task_id": task_id, "template_name": template_name, "reason": evaluation.reason})
                break
            ok, reason, passed, failed = validate_examples(utils, accepted_raw, task_id, -1)
            if not ok:
                failures.append({"task_id": task_id, "template_name": template_name, "reason": reason, "passed": passed, "failed": failed})
                break
            accepted = accepted_raw
            accepted_eval = asdict(evaluation)
            accepted_eval["validation_status"] = "full_arc_pass"
            accepted_eval["passed"] = passed
            accepted_eval["failed"] = failed
            break
        if accepted is None or accepted_eval is None:
            if not any(item["task_id"] == task_id for item in failures):
                failures.append({"task_id": task_id, "template_name": template_name, "reason": "template_not_found"})
            continue
        selected_raws[task_id] = accepted
        selected_rows.append(accepted_eval)

    bundle_raws = dict(raws)
    bundle_raws.update(selected_raws)
    zip_path = EXP_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_id in range(1, 401):
            zf.writestr(f"task{task_id:03d}.onnx", bundle_raws[task_id])

    delta = sum(float(r["candidate_points"]) - float(r["baseline_points"]) for r in selected_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "base_bundle": str(BASE_BUNDLE.relative_to(ROOT)),
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "selected_count": len(selected_rows),
        "selected_tasks": [int(r["task_id"]) for r in selected_rows],
        "selected_rows": selected_rows,
        "failures": failures,
        "local_delta": delta,
        "expected_public_lb_if_calibrated": BASE_PUBLIC_LB + delta,
        "zip_sanity": zip_sanity(zip_path),
        "submission_decision": "submit_ready" if selected_rows and not failures else "hold",
        "leakage_risk": "low-medium: full-arc gated graph surgery bundle stacked on exp259.",
        "overfitting_risk": "medium-low: private robustness requires LB calibration.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp260でfull-arc passしたrank141-220 bypass candidatesを、current best exp259 bundleに積んで提出zipを作成する。

## 結果

- selected_count: `{len(selected_rows)}`
- selected_tasks: `{[int(r["task_id"]) for r in selected_rows]}`
- failures: `{failures}`
- local_delta: `{delta}`
- expected_public_lb_if_calibrated: `{BASE_PUBLIC_LB + delta}`
- zip_sanity: `{result["zip_sanity"]}`

## 判断

`submission_decision` が `submit_ready` ならKaggleへ提出してbundle deltaを較正する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

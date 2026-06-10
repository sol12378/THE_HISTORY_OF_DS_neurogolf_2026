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
    point,
    score_model,
    validate_examples,
)


EXP_ID = "exp256_current_rank31_80_fullarc_bypass_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_BUNDLE = ROOT / "experiments" / "exp234_task300_max_color_submit_probe" / "submission.zip"
START_RANK = 31
END_RANK = 80
MAX_CANDIDATES_PER_TASK = 40


def load_zip_raws(path: pathlib.Path) -> dict[int, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def score_current_base(utils, raws: dict[int, bytes], proxy_base: dict[int, BaseTask], targets: list[int]) -> dict[int, BaseTask]:
    out: dict[int, BaseTask] = {}
    for task_id in targets:
        old = proxy_base[task_id]
        memory, params, reason = score_model(utils, raws[task_id], task_id, f"current_rank31_80_task{task_id:03d}", EXP_DIR)
        if memory is None or params is None:
            raise RuntimeError(f"score failed task{task_id:03d}: {reason}")
        cost = int(memory + params)
        out[task_id] = BaseTask(task_id, cost, point(cost), old.source, old.template_name, old.route, raws[task_id])
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    raws = load_zip_raws(BASE_BUNDLE)
    proxy_base = load_base_tasks()
    ranked = sorted(proxy_base.values(), key=lambda x: (-x.cost, x.task_id))
    targets = [item.task_id for item in ranked[START_RANK - 1 : END_RANK]]
    current_base = score_current_base(utils, raws, proxy_base, targets)

    eval_rows = []
    fullarc_rows = []
    selected_rows = []
    generated = 0

    for task_id in targets:
        task = current_base[task_id]
        best_eval = None
        cand_count = 0
        for candidate in e36.bypass_candidates(task_id, task.raw, task.route, 1):
            cand_count += 1
            if cand_count > MAX_CANDIDATES_PER_TASK:
                break
            generated += 1
            evaluation, accepted_raw = evaluate_candidate(utils, candidate, task, 20, EXP_DIR)
            eval_rows.append(asdict(evaluation))
            if accepted_raw is None:
                continue
            ok, reason, passed, failed = validate_examples(utils, accepted_raw, task_id, -1)
            fullarc_rows.append(
                {
                    "task_id": task_id,
                    "template_name": candidate.template_name,
                    "sample20_cost": evaluation.candidate_cost,
                    "sample20_points": evaluation.candidate_points,
                    "full_arc_ok": ok,
                    "passed": passed,
                    "failed": failed,
                    "reason": reason,
                    "sha256": evaluation.sha256,
                }
            )
            if ok and (best_eval is None or float(evaluation.candidate_points) > float(best_eval.candidate_points)):
                best_eval = evaluation
        if best_eval is not None:
            row = asdict(best_eval)
            row["validation_status"] = "full_arc_pass"
            selected_rows.append(row)

    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(eval_rows)
    with (EXP_DIR / "fullarc_candidate_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["task_id", "template_name", "sample20_cost", "sample20_points", "full_arc_ok", "passed", "failed", "reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fullarc_rows)
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields())
        writer.writeheader()
        writer.writerows(selected_rows)

    delta = sum(float(r["candidate_points"]) - float(r["baseline_points"]) for r in selected_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "base_bundle": str(BASE_BUNDLE.relative_to(ROOT)),
        "rank_window": [START_RANK, END_RANK],
        "max_candidates_per_task": MAX_CANDIDATES_PER_TASK,
        "targets": targets,
        "current_costs": {str(t): current_base[t].cost for t in targets},
        "generated_candidate_count": generated,
        "sample20_improved_count": len(fullarc_rows),
        "fullarc_selected_count": len(selected_rows),
        "selected_tasks": [int(r["task_id"]) for r in selected_rows],
        "selected_rows": selected_rows,
        "local_delta": delta,
        "submission_decision": "bundle_candidate" if selected_rows else "no_submit",
        "leakage_risk": "low-medium: full-arc gated graph surgery on current best bundle.",
        "overfitting_risk": "medium-low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp254の外側、current cost rank {START_RANK}-{END_RANK} に対して1-pass bypass surgeryをfull-arc gate付きで試す。

## 結果

- generated_candidate_count: `{generated}`
- sample20_improved_count: `{len(fullarc_rows)}`
- fullarc_selected_count: `{len(selected_rows)}`
- selected_tasks: `{[int(r["task_id"]) for r in selected_rows]}`
- local_delta: `{delta}`

## 判断

selectedがあればexp255の採点結果を見ながらbundle候補化する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

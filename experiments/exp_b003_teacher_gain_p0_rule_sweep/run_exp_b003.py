from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.exp_b002_p0_explainable_rule_sweep.run_exp_b002 import RULES, RuleResult, evaluate_rule  # noqa: E402


EXP_ID = "exp_b003_teacher_gain_p0_rule_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
EXP048_RESULT = ROOT / "experiments" / "exp048_submit_safe_seed_inventory" / "result.json"


def load_p0_tasks() -> list[dict[str, object]]:
    payload = json.loads(EXP048_RESULT.read_text(encoding="utf-8"))
    return list(payload["top_p0_tasks"])


def notes_text(result: dict[str, object]) -> str:
    return f"""# {EXP_ID}

## 目的

exp_b002ではqueue上位P0を対象にしてしまい、teacher-gain P0とずれていた。今回はexp048のteacher-gain P0 18 taskを対象に、同じ説明可能rule familyを評価する。

## 仮説

teacherが大きく改善したP0 taskは、queue上位signature lookupよりも説明可能rule hitが出やすい。

## 結果

- scanned tasks: {result["scanned_task_count"]}
- evaluated rows: {result["evaluated_rule_count"]}
- full pass hits: {result["full_pass_hit_count"]}
- train/test pass hits: {result["train_test_pass_hit_count"]}
- best partial: {result["best_partial"]}

## 解釈

full pass hitが出たら次はONNX lowering。出ない場合は、P0 sparse/object tasksにはD4/rectangle/lineより強いobject-role grammarが必要。
"""


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_p0_tasks()
    rows: list[RuleResult] = []
    for rank, item in enumerate(tasks, start=1):
        task_id = int(item["task_id"])
        family = str(item["route"])
        for rule_name, fn, lowering, risk in RULES:
            rows.append(evaluate_rule(task_id, rank, family, rule_name, fn, lowering, risk))
    rows = sorted(rows, key=lambda r: (r.status != "full_pass", r.status != "train_test_pass", -(r.total_pass / r.total_examples), r.rank, r.rule_name))
    full = [r for r in rows if r.status == "full_pass"]
    train_test = [r for r in rows if r.status == "train_test_pass"]
    best = rows[0] if rows else None
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "full_pass_found" if full else ("train_test_hit" if train_test else "no_full_hit"),
        "source": "experiments/exp048_submit_safe_seed_inventory top_p0_tasks",
        "scanned_task_count": len(tasks),
        "evaluated_rule_count": len(rows),
        "full_pass_hit_count": len(full),
        "train_test_pass_hit_count": len(train_test),
        "full_pass_hits": [asdict(r) for r in full[:20]],
        "train_test_hits": [asdict(r) for r in train_test[:20]],
        "best_partial": asdict(best) if best else None,
        "decision": "lower full_pass hits to ONNX; otherwise build object-role grammar from structural profiles",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no ONNX candidate generated",
        "leakage_risk": "low: no teacher label table or arc-gen memorization.",
        "overfitting_risk": "medium: all arc-gen used for diagnostics; accepted candidates need family holdout and Kaggle delta submission.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

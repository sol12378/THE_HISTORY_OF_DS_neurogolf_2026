from __future__ import annotations

import csv
import json
import pathlib
import sys
from dataclasses import asdict
from datetime import date

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import Candidate, evaluate_candidate, load_base_tasks, load_neurogolf_utils  # noqa: E402


EXP_ID = "exp251_small_output_prune_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGETS = [100, 242, 253, 271]


def prune_unused_initializers(raw: bytes) -> tuple[bytes, int]:
    model = onnx.load_model_from_string(raw)
    used = set()
    for node in model.graph.node:
        used.update(name for name in node.input if name)
    graph_inputs = {value.name for value in model.graph.input}
    keep = []
    removed = 0
    for init in model.graph.initializer:
        if init.name in used or init.name in graph_inputs:
            keep.append(init)
        else:
            removed += 1
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
    return model.SerializeToString(), removed


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    bases = load_base_tasks()
    rows = []
    results = []
    for task_id in TARGETS:
        base = bases[task_id]
        raw, removed = prune_unused_initializers(base.raw)
        candidate = Candidate(
            task_id,
            f"unused_initializer_prune_removed_{removed}",
            base.route,
            raw,
            "generated",
            f"remove unused initializers from current baseline; removed={removed}",
        )
        eval_row, improved_raw = evaluate_candidate(utils, candidate, base, -1, EXP_DIR)
        row = asdict(eval_row)
        row["removed_initializers"] = removed
        rows.append(row)
        results.append({"task_id": task_id, "removed": removed, "eval": row, "improved": improved_raw is not None})

    with (EXP_DIR / "prune_probe.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "targets": TARGETS,
        "results": results,
        "improved_tasks": [r["task_id"] for r in results if r["improved"]],
        "decision": "If any improved, consider bundle/submission delta. Otherwise move to richer surgery or new candidate mining.",
        "submission_decision": "no_submit: probe only",
        "leakage_risk": "low: semantics-preserving graph cleanup.",
        "overfitting_risk": "low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

small-output solved/near-solved候補 task100/242/253/271 の既存artifactに未使用initializerが残っていれば削除し、score改善するか確認する。

## 結果

- improved_tasks: `{result['improved_tasks']}`
- rows: `{rows}`

## 判断

改善があればbundle/submission候補。なければより深いsurgeryまたは新規候補探索へ戻る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

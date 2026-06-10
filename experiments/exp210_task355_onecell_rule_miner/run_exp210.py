from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp210_task355_onecell_rule_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 355


def nonzero_values(x: np.ndarray) -> list[int]:
    return [int(v) for v in x.ravel() if int(v) != 0]


def color_counts(x: np.ndarray, include_zero: bool = False) -> Counter[int]:
    vals = [int(v) for v in x.ravel() if include_zero or int(v) != 0]
    return Counter(vals)


def mode_color(x: np.ndarray) -> int:
    counts = color_counts(x)
    return min(counts, key=lambda c: (-counts[c], c)) if counts else 0


def least_color(x: np.ndarray) -> int:
    counts = color_counts(x)
    return min(counts, key=lambda c: (counts[c], c)) if counts else 0


def unique_color(x: np.ndarray) -> int:
    counts = color_counts(x)
    uniques = [c for c, n in counts.items() if n == 1]
    return min(uniques) if uniques else 0


def border_mode(x: np.ndarray) -> int:
    vals = np.concatenate([x[0, :], x[-1, :], x[1:-1, 0], x[1:-1, -1]])
    counts = Counter(int(v) for v in vals if int(v) != 0)
    return min(counts, key=lambda c: (-counts[c], c)) if counts else 0


def inner_mode(x: np.ndarray) -> int:
    if x.shape[0] <= 2 or x.shape[1] <= 2:
        return mode_color(x)
    counts = Counter(int(v) for v in x[1:-1, 1:-1].ravel() if int(v) != 0)
    return min(counts, key=lambda c: (-counts[c], c)) if counts else 0


def center_color(x: np.ndarray) -> int:
    return int(x[x.shape[0] // 2, x.shape[1] // 2])


def corner_tl(x: np.ndarray) -> int:
    return int(x[0, 0])


def corner_tr(x: np.ndarray) -> int:
    return int(x[0, -1])


def corner_bl(x: np.ndarray) -> int:
    return int(x[-1, 0])


def corner_br(x: np.ndarray) -> int:
    return int(x[-1, -1])


def bbox_crop(x: np.ndarray) -> np.ndarray:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return x
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return x[int(r0) : int(r1), int(c0) : int(c1)]


def bbox_center(x: np.ndarray) -> int:
    b = bbox_crop(x)
    return center_color(b)


def bbox_mode(x: np.ndarray) -> int:
    return mode_color(bbox_crop(x))


def bbox_least(x: np.ndarray) -> int:
    return least_color(bbox_crop(x))


def output_color(ex: dict[str, Any]) -> int:
    y = grid_to_array(ex["output"])
    return int(y[0, 0])


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    base = load_base_tasks()[TASK_ID]
    funcs: list[tuple[str, Callable[[np.ndarray], int]]] = [
        ("mode_color", mode_color),
        ("least_color", least_color),
        ("unique_color", unique_color),
        ("border_mode", border_mode),
        ("inner_mode", inner_mode),
        ("center_color", center_color),
        ("corner_tl", corner_tl),
        ("corner_tr", corner_tr),
        ("corner_bl", corner_bl),
        ("corner_br", corner_br),
        ("bbox_center", bbox_center),
        ("bbox_mode", bbox_mode),
        ("bbox_least", bbox_least),
    ]
    rows: list[dict[str, Any]] = []
    fail_examples: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in funcs}
    pass_counts = Counter()
    output_hist: Counter[int] = Counter()
    shape_hist: Counter[str] = Counter()
    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = output_color(ex)
        output_hist[y] += 1
        shape_hist[f"{x.shape[0]}x{x.shape[1]}"] += 1
        row = {"idx": idx, "shape": f"{x.shape[0]}x{x.shape[1]}", "output": y}
        for name, fn in funcs:
            pred = int(fn(x))
            ok = pred == y
            pass_counts[name] += int(ok)
            row[name] = pred
            if not ok and len(fail_examples[name]) < 8:
                fail_examples[name].append({"idx": idx, "shape": row["shape"], "pred": pred, "expected": y})
        rows.append(row)

    with (EXP_DIR / "rule_miner_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(
        [{"rule": name, "pass_count": int(pass_counts[name]), "fail_count": len(examples) - int(pass_counts[name])} for name, _ in funcs],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    best = ranked[0]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if best["pass_count"] == len(examples) else "partial",
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "baseline_points": base.points,
        "example_count": len(examples),
        "shape_hist": dict(sorted(shape_hist.items())),
        "output_hist": dict(sorted(output_hist.items())),
        "ranked_rules": ranked,
        "best_rule": best,
        "best_fail_examples": fail_examples[best["rule"]],
        "decision": "If a simple 1-cell rule is full-pass, lower it with a tiny ONNX reducer; otherwise inspect near-miss features or pivot.",
        "submission_decision": "no_submit: Python rule mining only",
        "leakage_risk": "low: input-only aggregate rules.",
        "overfitting_risk": "medium-low for simple aggregates, higher if adding branch tables.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp087の1x1 cropish上位task355について、単純な入力集約ruleでoutput colorを説明できるか高速に確認する。

## 結果

- best_rule: `{best['rule']}`
- pass: `{best['pass_count']}/{len(examples)}`
- output_hist: `{dict(sorted(output_hist.items()))}`

## 判断

full passならtiny ONNX reducerへ進む。partialならnear-miss特徴を見て、短く伸ばせないなら別taskへpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

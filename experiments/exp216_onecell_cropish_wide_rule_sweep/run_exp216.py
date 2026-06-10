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


EXP_ID = "exp216_onecell_cropish_wide_rule_sweep"
EXP_DIR = ROOT / "experiments" / EXP_ID
CANDIDATE_CSV = ROOT / "experiments" / "exp087_small_output_crop_candidate_scan" / "small_output_candidates.csv"


def counts(x: np.ndarray, include_zero: bool = False) -> Counter[int]:
    return Counter(int(v) for v in x.ravel() if include_zero or int(v) != 0)


def choose_count(x: np.ndarray, reverse: bool, include_zero: bool = False, rank: int = 0) -> int:
    c = counts(x, include_zero)
    if not c:
        return 0
    ordered = sorted(c, key=lambda k: ((-c[k] if reverse else c[k]), k))
    return int(ordered[min(rank, len(ordered) - 1)])


def bbox_crop(x: np.ndarray) -> np.ndarray:
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return x
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0) + 1
    return x[int(r0) : int(r1), int(c0) : int(c1)]


def middle_row(x: np.ndarray) -> np.ndarray:
    return x[x.shape[0] // 2 : x.shape[0] // 2 + 1, :]


def middle_col(x: np.ndarray) -> np.ndarray:
    return x[:, x.shape[1] // 2 : x.shape[1] // 2 + 1]


def corners(x: np.ndarray) -> np.ndarray:
    return np.array([x[0, 0], x[0, -1], x[-1, 0], x[-1, -1]], dtype=x.dtype)


def edge_cells(x: np.ndarray) -> np.ndarray:
    if min(x.shape) == 1:
        return x.ravel()
    return np.concatenate([x[0, :], x[-1, :], x[1:-1, 0], x[1:-1, -1]])


def interior_cells(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 2 or x.shape[1] <= 2:
        return np.array([], dtype=x.dtype)
    return x[1:-1, 1:-1].ravel()


def unique_nz_or_zero(x: np.ndarray) -> int:
    vals = sorted({int(v) for v in x.ravel() if int(v) != 0})
    return int(vals[0]) if len(vals) == 1 else 0


def funcs() -> list[tuple[str, Callable[[np.ndarray], int]]]:
    bases: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
        ("all", lambda x: x),
        ("bbox", bbox_crop),
        ("middle_row", middle_row),
        ("middle_col", middle_col),
        ("corners", corners),
        ("edge", edge_cells),
        ("interior", interior_cells),
    ]
    out: list[tuple[str, Callable[[np.ndarray], int]]] = [
        ("center", lambda x: int(x[x.shape[0] // 2, x.shape[1] // 2])),
        ("tl", lambda x: int(x[0, 0])),
        ("tr", lambda x: int(x[0, -1])),
        ("bl", lambda x: int(x[-1, 0])),
        ("br", lambda x: int(x[-1, -1])),
        ("unique_nz", unique_nz_or_zero),
        ("bbox_center", lambda x: int((b := bbox_crop(x))[b.shape[0] // 2, b.shape[1] // 2])),
    ]
    for prefix, base_fn in bases:
        out.extend(
            [
                (f"{prefix}_mode_nz", lambda x, fn=base_fn: choose_count(fn(x), True)),
                (f"{prefix}_least_nz", lambda x, fn=base_fn: choose_count(fn(x), False)),
                (f"{prefix}_second_least_nz", lambda x, fn=base_fn: choose_count(fn(x), False, False, 1)),
                (f"{prefix}_mode_all", lambda x, fn=base_fn: choose_count(fn(x), True, True)),
                (f"{prefix}_least_all", lambda x, fn=base_fn: choose_count(fn(x), False, True)),
            ]
        )
    return out


def output_color(ex: dict[str, Any]) -> int:
    y = grid_to_array(ex["output"])
    return int(y[0, 0])


def load_candidate_task_ids() -> list[int]:
    task_ids: list[int] = []
    with CANDIDATE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["is_cropish"] != "True":
                continue
            if int(row["max_output_area"]) != 1 or int(row["min_output_area"]) != 1:
                continue
            task_ids.append(int(row["task_id"]))
    return sorted(set(task_ids))


def is_all_onecell(task: dict[str, Any]) -> bool:
    examples = task["train"] + task["test"] + task["arc-gen"]
    return all(grid_to_array(ex["output"]).shape == (1, 1) for ex in examples)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()
    task_ids = load_candidate_task_ids()
    rules = funcs()
    rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for task_id in task_ids:
        task = load_task(task_id)
        if not is_all_onecell(task):
            skipped.append({"task_id": task_id, "reason": "not_all_outputs_1x1"})
            continue
        examples = task["train"] + task["test"] + task["arc-gen"]
        pass_counts = Counter()
        output_hist = Counter(output_color(ex) for ex in examples)
        for ex in examples:
            x = grid_to_array(ex["input"])
            y = output_color(ex)
            for name, fn in rules:
                try:
                    pred = int(fn(x))
                except Exception:
                    pred = -999
                pass_counts[name] += int(pred == y)
        ranked = sorted(
            [
                {
                    "rule": name,
                    "pass_count": int(pass_counts[name]),
                    "fail_count": len(examples) - int(pass_counts[name]),
                }
                for name, _ in rules
            ],
            key=lambda r: (-r["pass_count"], r["fail_count"], r["rule"]),
        )
        best = ranked[0]
        result_rows.append(
            {
                "task_id": task_id,
                "baseline_cost": base[task_id].cost,
                "baseline_points": base[task_id].points,
                "example_count": len(examples),
                "output_hist": dict(sorted(output_hist.items())),
                "best_rule": best["rule"],
                "best_pass_count": best["pass_count"],
                "best_fail_count": best["fail_count"],
                "top_rules": ranked[:10],
            }
        )
        for rank, item in enumerate(ranked[:20]):
            rows.append({"task_id": task_id, "rank": rank, **item})

    with (EXP_DIR / "onecell_wide_rule_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "rank", "rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(rows)

    full_hits = [r for r in result_rows if r["best_fail_count"] == 0]
    near_hits = [r for r in result_rows if 0 < r["best_fail_count"] <= 10]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else ("near_hit" if near_hits else "no_full_hit"),
        "candidate_task_ids": task_ids,
        "evaluated_task_ids": [r["task_id"] for r in result_rows],
        "skipped": skipped,
        "rule_count": len(rules),
        "rows": result_rows,
        "full_hits": full_hits,
        "near_hits": near_hits,
        "decision": "Full hitはtiny ONNX cost probeへ進める。near hitは失敗監査。全滅なら1x1単純集約laneを縮小し、別familyへpivotする。",
        "submission_decision": "no_submit: rule sweep only",
        "leakage_risk": "low: input-only aggregate rules, no train-output lookup.",
        "overfitting_risk": "medium-low: simple hand-written aggregators only; near-hit branch追加時は再評価が必要。",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp087の1x1 cropish候補全体へ、安くloweringできる可能性のある入力集約ruleを横展開する。

## 仮説

1x1出力taskには、mode/least/corner/bbox/edge/interiorのような小さな集約で説明できるものが残っている。full hitが見つかれば、one-cell scalar graphとして低cost化できる可能性がある。

## 結果

- evaluated: `{[r['task_id'] for r in result_rows]}`
- full_hits: `{[(r['task_id'], r['best_rule']) for r in full_hits]}`
- near_hits(<=10 fail): `{[(r['task_id'], r['best_rule'], r['best_fail_count']) for r in near_hits]}`
- best: `{[(r['task_id'], r['best_rule'], r['best_pass_count'], r['example_count']) for r in result_rows]}`

## 判断

full hitがあればtiny ONNX cost probeへ進める。near hitは失敗監査を行う。full/near hitが薄ければ、1x1単純集約laneは優先度を下げ、component-freeな別small-output familyへ移る。

## リスク

- leakage risk: low。入力だけの集約rule。
- overfitting risk: medium-low。単純ruleのため低いが、near hit補正で分岐を足す場合は上がる。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

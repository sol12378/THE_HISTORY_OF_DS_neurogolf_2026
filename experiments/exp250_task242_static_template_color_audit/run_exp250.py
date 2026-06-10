from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp250_task242_static_template_color_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 242


def output_color(y: np.ndarray) -> int:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def binary_template(y: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v != 0) for v in y.ravel())


def largest_component_size(x: np.ndarray, color: int) -> int:
    seen = np.zeros(x.shape, dtype=bool)
    best = 0
    for r, c in np.argwhere(x == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        size = 0
        while q:
            rr, cc = q.popleft()
            size += 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1] and not seen[nr, nc] and int(x[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        best = max(best, size)
    return best


def color_stats(x: np.ndarray) -> list[dict[str, Any]]:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    rows = []
    for color, count in sorted(counts.items()):
        cells = np.argwhere(x == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        rows.append(
            {
                "color": int(color),
                "count": int(count),
                "bbox_h": int(r1 - r0 + 1),
                "bbox_w": int(c1 - c0 + 1),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "largest_component": largest_component_size(x, int(color)),
            }
        )
    return rows


def candidate_rules(stats: list[dict[str, Any]]) -> dict[str, int]:
    rules: dict[str, int] = {}
    if not stats:
        return rules
    for key in ("count", "bbox_h", "bbox_w", "bbox_area", "r0", "c0", "r1", "c1", "largest_component"):
        rules[f"{key}_max_low"] = sorted(stats, key=lambda r: (-r[key], r["color"]))[0]["color"]
        rules[f"{key}_max_high"] = sorted(stats, key=lambda r: (-r[key], -r["color"]))[0]["color"]
        rules[f"{key}_min_low"] = sorted(stats, key=lambda r: (r[key], r["color"]))[0]["color"]
        rules[f"{key}_min_high"] = sorted(stats, key=lambda r: (r[key], -r["color"]))[0]["color"]
    count_rank = sorted(stats, key=lambda r: (-r["count"], r["color"]))
    for i, row in enumerate(count_rank[:5]):
        rules[f"count_rank{i}"] = row["color"]
    area_rank = sorted(stats, key=lambda r: (-r["bbox_area"], r["color"]))
    for i, row in enumerate(area_rank[:5]):
        rules[f"area_rank{i}"] = row["color"]
    return rules


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]
    rows: list[dict[str, Any]] = []
    template_hist = Counter()
    color_hist = Counter()
    rule_hits = Counter()
    target_count_rank_hist = Counter()
    target_area_rank_hist = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        target = output_color(y)
        template_hist[binary_template(y)] += 1
        color_hist[target] += 1
        stats = color_stats(x)
        rules = candidate_rules(stats)
        for name, pred in rules.items():
            rule_hits[name] += int(pred == target)
        count_rank = sorted(stats, key=lambda r: (-r["count"], r["color"]))
        area_rank = sorted(stats, key=lambda r: (-r["bbox_area"], r["color"]))
        target_count_rank_hist[next((i for i, row in enumerate(count_rank) if row["color"] == target), -1)] += 1
        target_area_rank_hist[next((i for i, row in enumerate(area_rank) if row["color"] == target), -1)] += 1
        rows.append({"idx": idx, "target": target, "stats": stats, "rules_hit": [k for k, v in rules.items() if v == target]})

    ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in rule_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    with (EXP_DIR / "static_template_color_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "target", "stats", "rules_hit"])
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "rule_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(ranked)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "template_hist": [[str(k), int(v)] for k, v in template_hist.most_common()],
        "output_color_hist": dict(sorted(color_hist.items())),
        "target_count_rank_hist": dict(sorted(target_count_rank_hist.items())),
        "target_area_rank_hist": dict(sorted(target_area_rank_hist.items())),
        "best_rules": ranked[:40],
        "decision": "If a simple color selector is full, build static 3x3 ONNX probe. Otherwise hold task242.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task242は3x3固定・binary template 1種類・baseline高めなので、色selectorをcount/bbox/component特徴で監査する。

## 結果

- baseline_cost: `{base.cost}`
- template_hist: `{template_hist.most_common()}`
- output_color_hist: `{dict(sorted(color_hist.items()))}`
- target_count_rank_hist: `{dict(sorted(target_count_rank_hist.items()))}`
- target_area_rank_hist: `{dict(sorted(target_area_rank_hist.items()))}`
- best_rules: `{ranked[:10]}`

## 判断

単純color selectorがfullならstatic 3x3 ONNX probeへ進む。弱ければ保留。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

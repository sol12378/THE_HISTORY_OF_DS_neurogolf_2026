from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import grid_to_array, load_base_tasks, load_task  # noqa: E402


EXP_ID = "exp246_task100_mode_failure_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 100


def output_color(y: np.ndarray) -> int:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def color_stats(x: np.ndarray) -> list[dict[str, Any]]:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    rows = []
    for color, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        cells = np.argwhere(x == color)
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        on_edge = int(np.any((cells[:, 0] == 0) | (cells[:, 0] == x.shape[0] - 1) | (cells[:, 1] == 0) | (cells[:, 1] == x.shape[1] - 1)))
        rows.append(
            {
                "color": int(color),
                "count": int(count),
                "rank": len(rows),
                "bbox_h": int(r1 - r0 + 1),
                "bbox_w": int(c1 - c0 + 1),
                "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "on_edge": on_edge,
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
            }
        )
    return rows


def candidate_rules(stats: list[dict[str, Any]]) -> dict[str, int]:
    rules: dict[str, int] = {}
    if not stats:
        return rules
    for key in ("count", "bbox_h", "bbox_w", "bbox_area", "on_edge", "r0", "c0", "r1", "c1"):
        rules[f"{key}_max"] = sorted(stats, key=lambda r: (-r[key], r["color"]))[0]["color"]
        rules[f"{key}_min"] = sorted(stats, key=lambda r: (r[key], r["color"]))[0]["color"]
    for rank in range(min(4, len(stats))):
        rules[f"rank{rank}"] = stats[rank]["color"]
    # The exp245 default.
    rules["mode_low"] = sorted(stats, key=lambda r: (-r["count"], r["color"]))[0]["color"]
    rules["mode_high"] = sorted(stats, key=lambda r: (-r["count"], -r["color"]))[0]["color"]
    return rules


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    rule_hits = Counter()
    target_rank_hist = Counter()
    failure_rows = []

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        target = output_color(y)
        stats = color_stats(x)
        rules = candidate_rules(stats)
        for name, pred in rules.items():
            rule_hits[name] += int(pred == target)
        target_stat = next((s for s in stats if s["color"] == target), {})
        if target_stat:
            target_rank_hist[target_stat["rank"]] += 1
        mode_low = rules.get("mode_low", 0)
        rec = {
            "idx": idx,
            "target": target,
            "mode_low": mode_low,
            "mode_ok": mode_low == target,
            "target_stat": target_stat,
            "stats": stats,
            "rules_hit": [name for name, pred in rules.items() if pred == target],
        }
        rows.append(rec)
        if mode_low != target:
            failure_rows.append(rec)

    ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in rule_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    with (EXP_DIR / "mode_failure_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "target", "mode_low", "mode_ok", "target_stat", "stats", "rules_hit"])
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
        "mode_low_pass": int(rule_hits["mode_low"]),
        "mode_low_fail": len(examples) - int(rule_hits["mode_low"]),
        "target_rank_hist": dict(sorted(target_rank_hist.items())),
        "best_rules": ranked[:30],
        "failure_rows": failure_rows[:30],
        "decision": "If a simple rank/geometric color selector is full, proceed to Python rule and cost probe. Otherwise hold task100.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task100のmode_nz rule `248/266` の残り18例を監査し、count rankや色bbox特徴で切替条件が作れるか確認する。

## 結果

- baseline_cost: `{base.cost}`
- mode_low: `{int(rule_hits['mode_low'])}/{len(examples)}`
- target_rank_hist: `{dict(sorted(target_rank_hist.items()))}`
- best_rules: `{ranked[:10]}`

## 判断

単純selectorがfullならstatic 2x2 ONNXへ進む。なければtask100は保留。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

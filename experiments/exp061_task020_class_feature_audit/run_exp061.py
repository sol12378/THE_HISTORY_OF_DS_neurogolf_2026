from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402
from experiments.exp059_task020_three_template_selector.run_exp059 import input_features, true_template_name  # noqa: E402


EXP_ID = "exp061_task020_class_feature_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class KeySummary:
    feature_key: str
    unique_key_count: int
    pure_key_count: int
    covered_examples: int
    majority_correct: int
    accuracy: float
    top_conflicts: str


@dataclass(frozen=True)
class ExampleFeature:
    example_idx: int
    split: str
    template_name: str
    target_color: int
    feature_key: str
    feature_value: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def scalar_features(features: dict[str, Any]) -> dict[str, Any]:
    pos = tuple(features["pos"])
    canon = tuple(features["canon_pos"])
    row_counts = tuple(features["row_counts"])
    col_counts = tuple(features["col_counts"])
    return {
        "pos": pos,
        "canon_pos": canon,
        "pos_count": features["pos_count"],
        "row_counts": row_counts,
        "col_counts": col_counts,
        "sorted_row_counts": tuple(sorted(row_counts, reverse=True)),
        "sorted_col_counts": tuple(sorted(col_counts, reverse=True)),
        "touches_center": features["touches_center"],
        "touches_corner": features["touches_corner"],
        "touches_edge_mid": features["touches_edge_mid"],
        "touch_signature": (
            bool(features["touches_center"]),
            bool(features["touches_corner"]),
            bool(features["touches_edge_mid"]),
        ),
        "count_and_touch": (
            features["pos_count"],
            bool(features["touches_center"]),
            bool(features["touches_corner"]),
            bool(features["touches_edge_mid"]),
        ),
        "canon_and_count": (canon, features["pos_count"]),
        "row_col_counts": (row_counts, col_counts),
    }


def summarize_key(records: list[dict[str, Any]], key: str) -> KeySummary:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        table[json.dumps(rec["features"][key], sort_keys=True, default=str)][rec["template_name"]] += 1
    majority_correct = 0
    pure = 0
    conflicts = []
    for value, cnt in table.items():
        majority_correct += cnt.most_common(1)[0][1]
        if len(cnt) == 1:
            pure += 1
        else:
            conflicts.append((sum(cnt.values()), value, dict(cnt)))
    conflicts.sort(reverse=True)
    total = len(records)
    return KeySummary(
        feature_key=key,
        unique_key_count=len(table),
        pure_key_count=pure,
        covered_examples=total,
        majority_correct=majority_correct,
        accuracy=majority_correct / total,
        top_conflicts=json.dumps(conflicts[:8], ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    records: list[dict[str, Any]] = []
    example_rows: list[ExampleFeature] = []
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        template_name, _, color = true_template_name(x, y)
        feats = scalar_features(input_features(x, color))
        rec = {
            "idx": idx,
            "split": split_name(idx, train_n, test_n),
            "template_name": template_name,
            "target_color": color,
            "features": feats,
        }
        records.append(rec)
        for key, value in feats.items():
            example_rows.append(
                ExampleFeature(
                    example_idx=idx,
                    split=rec["split"],
                    template_name=template_name,
                    target_color=color,
                    feature_key=key,
                    feature_value=json.dumps(value, ensure_ascii=False, sort_keys=True, default=str),
                )
            )

    key_rows = [summarize_key(records, key) for key in records[0]["features"]]
    key_rows.sort(key=lambda r: (r.accuracy, r.pure_key_count, -r.unique_key_count), reverse=True)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_ready",
        "target_task": TARGET_TASK,
        "example_count": len(records),
        "template_counts": dict(Counter(r["template_name"] for r in records)),
        "best_key": asdict(key_rows[0]),
        "key_summaries": [asdict(r) for r in key_rows],
        "decision": "If a low-cardinality high-accuracy key exists, turn it into a generative selector; avoid high-cardinality lookup selectors.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic audit only",
        "leakage_risk": "medium: all examples used for feature audit.",
        "overfitting_risk": "high for high-cardinality keys; acceptable only if reduced to simple rule.",
        "outputs": {
            "key_summary": "key_summary.csv",
            "example_features": "example_features.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "key_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(KeySummary.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in key_rows])
    with (EXP_DIR / "example_features.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleFeature.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in example_rows])
    notes = f"""# {EXP_ID}

## 目的

task020の3 template classを入力特徴から選ぶため、feature keyごとの純度を調べる。

## 結果

- examples: {len(records)}
- template counts: {dict(Counter(r['template_name'] for r in records))}
- best key: {asdict(key_rows[0])}

## Key Summary

| key | accuracy | unique keys | pure keys |
|---|---:|---:|---:|
"""
    for row in key_rows[:12]:
        notes += f"| {row.feature_key} | {row.accuracy:.3f} | {row.unique_key_count} | {row.pure_key_count} |\n"
    notes += """
## Decision

高cardinality lookupは使わない。低cardinalityで高精度な特徴があれば、rule selectorへ落とす。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

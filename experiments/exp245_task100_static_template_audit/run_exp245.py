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


EXP_ID = "exp245_task100_static_template_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 100


def normalize_template(y: np.ndarray) -> tuple[int, ...]:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    mode = Counter(vals).most_common(1)[0][0] if vals else 0
    return tuple(int(v == mode) for v in y.ravel())


def output_color(y: np.ndarray) -> int:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    return Counter(vals).most_common(1)[0][0] if vals else 0


def input_features(x: np.ndarray) -> dict[str, int]:
    nz_counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    all_counts = Counter(int(v) for v in x.ravel())
    feats: dict[str, int] = {
        "mode_nz": min(nz_counts, key=lambda k: (-nz_counts[k], k)) if nz_counts else 0,
        "least_nz": min(nz_counts, key=lambda k: (nz_counts[k], k)) if nz_counts else 0,
        "mode_all": min(all_counts, key=lambda k: (-all_counts[k], k)),
        "center": int(x[x.shape[0] // 2, x.shape[1] // 2]),
        "tl": int(x[0, 0]),
        "tr": int(x[0, -1]),
        "bl": int(x[-1, 0]),
        "br": int(x[-1, -1]),
    }
    nz = np.argwhere(x != 0)
    if len(nz):
        r0, c0 = nz.min(axis=0)
        r1, c1 = nz.max(axis=0)
        feats.update(
            {
                "bbox_tl": int(x[int(r0), int(c0)]),
                "bbox_tr": int(x[int(r0), int(c1)]),
                "bbox_bl": int(x[int(r1), int(c0)]),
                "bbox_br": int(x[int(r1), int(c1)]),
                "bbox_center": int(x[(int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2]),
            }
        )
    # Per-color geometric anchors.
    for color in range(1, 10):
        cells = np.argwhere(x == color)
        if len(cells):
            r0, c0 = cells.min(axis=0)
            r1, c1 = cells.max(axis=0)
            feats[f"color{color}_count"] = int(len(cells))
            feats[f"color{color}_bbox_h"] = int(r1 - r0 + 1)
            feats[f"color{color}_bbox_w"] = int(c1 - c0 + 1)
    return feats


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    template_hist = Counter()
    color_hist = Counter()
    feature_color_hits = Counter()
    max_count_color_hits = Counter()
    min_count_color_hits = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        tpl = normalize_template(y)
        color = output_color(y)
        template_hist[tpl] += 1
        color_hist[color] += 1
        feats = input_features(x)
        for name, value in feats.items():
            feature_color_hits[name] += int(value == color)
        nz_counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
        for tie in ("low", "high"):
            if nz_counts:
                max_color = sorted(nz_counts, key=lambda k: (-nz_counts[k], k if tie == "low" else -k))[0]
                min_color = sorted(nz_counts, key=lambda k: (nz_counts[k], k if tie == "low" else -k))[0]
                max_count_color_hits[tie] += int(max_color == color)
                min_count_color_hits[tie] += int(min_color == color)
        rows.append({"idx": idx, "output_color": color, "template": tpl, "features": feats})

    ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in feature_color_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    ranked.extend(
        [
            {"rule": f"max_count_{tie}", "pass_count": int(v), "fail_count": len(examples) - int(v)}
            for tie, v in max_count_color_hits.items()
        ]
    )
    ranked.extend(
        [
            {"rule": f"min_count_{tie}", "pass_count": int(v), "fail_count": len(examples) - int(v)}
            for tie, v in min_count_color_hits.items()
        ]
    )
    ranked = sorted(ranked, key=lambda r: (-r["pass_count"], r["rule"]))

    with (EXP_DIR / "static_template_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "output_color", "template", "features"])
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "color_rule_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule", "pass_count", "fail_count"])
        writer.writeheader()
        writer.writerows(ranked)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "output_shape": list(grid_to_array(examples[0]["output"]).shape),
        "template_hist": [[str(k), int(v)] for k, v in template_hist.most_common()],
        "output_color_hist": dict(sorted(color_hist.items())),
        "best_color_rules": ranked[:30],
        "decision": "If a simple color rule is full, build static 2x2 template ONNX. Otherwise hold task100.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium-low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task100は2x2固定・binary signature 1種類のstable-binary候補なので、出力templateと色selectorを監査する。

## 結果

- baseline_cost: `{base.cost}`
- template_hist: `{template_hist.most_common()}`
- output_color_hist: `{dict(sorted(color_hist.items()))}`
- best_color_rules: `{ranked[:10]}`

## 判断

単純color ruleがfullならstatic 2x2 ONNXへ進む。なければ保留。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

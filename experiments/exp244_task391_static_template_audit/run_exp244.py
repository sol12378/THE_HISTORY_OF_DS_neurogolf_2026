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


EXP_ID = "exp244_task391_static_template_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 391


def output_signature(y: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    vals = [int(v) for v in y.ravel() if int(v) != 0]
    mode = Counter(vals).most_common(1)[0][0] if vals else 0
    binary = tuple(int(v != 0) for v in y.ravel())
    colorized = tuple(int(v == mode) for v in y.ravel())
    return binary, colorized


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
    return feats


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks()[TASK_ID]
    task = load_task(TASK_ID)
    examples = task["train"] + task["test"] + task["arc-gen"]

    rows: list[dict[str, Any]] = []
    binary_hist = Counter()
    colorized_hist = Counter()
    out_color_hist = Counter()
    feature_color_hits = Counter()
    feature_template_hits = Counter()
    joint_hits = Counter()

    for idx, ex in enumerate(examples):
        x = grid_to_array(ex["input"])
        y = grid_to_array(ex["output"])
        binary, colorized = output_signature(y)
        vals = [int(v) for v in y.ravel() if int(v) != 0]
        out_color = Counter(vals).most_common(1)[0][0] if vals else 0
        binary_hist[binary] += 1
        colorized_hist[colorized] += 1
        out_color_hist[out_color] += 1
        feats = input_features(x)
        for name, value in feats.items():
            feature_color_hits[name] += int(value == out_color)
        # If a feature is itself one of 0/1/2, try it as a binary template id after sorting templates by frequency.
        rows.append({"idx": idx, "out_color": out_color, "binary": binary, "colorized": colorized, "features": feats})

    templates = [tpl for tpl, _ in binary_hist.most_common()]
    template_id = {tpl: i for i, tpl in enumerate(templates)}
    for row in rows:
        tid = template_id[row["binary"]]
        for name, value in row["features"].items():
            feature_template_hits[name] += int(value == tid)
            joint_hits[f"{name}_color_and_template"] += int(value == row["out_color"] and value == tid)

    color_ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in feature_color_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    template_ranked = sorted(
        [{"rule": k, "pass_count": int(v), "fail_count": len(examples) - int(v)} for k, v in feature_template_hits.items()],
        key=lambda r: (-r["pass_count"], r["rule"]),
    )
    with (EXP_DIR / "static_template_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "out_color", "binary", "colorized", "features"])
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "rule_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "rule", "pass_count", "fail_count"])
        writer.writeheader()
        for r in color_ranked:
            writer.writerow({"kind": "color", **r})
        for r in template_ranked:
            writer.writerow({"kind": "template", **r})

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "baseline_cost": base.cost,
        "example_count": len(examples),
        "output_shape": list(grid_to_array(examples[0]["output"]).shape),
        "binary_signature_count": len(binary_hist),
        "binary_hist": [[str(k), int(v)] for k, v in binary_hist.most_common()],
        "colorized_signature_count": len(colorized_hist),
        "output_color_hist": dict(sorted(out_color_hist.items())),
        "best_color_rules": color_ranked[:20],
        "best_template_rules": template_ranked[:20],
        "decision": "If simple color and template selectors are full, build static 3x1 ONNX. Otherwise task391 needs richer selector.",
        "submission_decision": "no_submit: audit only",
        "leakage_risk": "low.",
        "overfitting_risk": "medium-low.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task391は3x1固定出力・binary signature 3種類のstable-binary候補なので、出力templateと色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline_cost: `{base.cost}`
- binary_signature_count: `{len(binary_hist)}`
- binary_hist: `{binary_hist.most_common()}`
- output_color_hist: `{dict(sorted(out_color_hist.items()))}`
- best_color_rules: `{color_ranked[:10]}`
- best_template_rules: `{template_ranked[:10]}`

## 判断

単純selectorがfullならstatic 3x1 ONNXへ進む。なければ別候補へpivotする。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

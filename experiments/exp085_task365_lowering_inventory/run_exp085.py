from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402
from experiments.exp084_task365_object_crop_rule_probe.run_exp084 import arr, object_patches  # noqa: E402


EXP_ID = "exp085_task365_lowering_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 365


@dataclass(frozen=True)
class InventoryRow:
    idx: int
    split: str
    object_count: int
    selected_rank: int
    selected_bbox: str
    selected_shape: str
    max_count2: int
    count2_hist: str
    all_objects_dense_rectangles: int
    input_shape: str
    output_shape: str


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    rows: list[InventoryRow] = []
    object_counts = Counter()
    selected_shapes = Counter()
    max_count2_hist = Counter()
    dense_ok = 0
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        objs = object_patches(x)
        object_counts[len(objs)] += 1
        count2_values = [int(o["count2"]) for o in objs]
        max_count2 = max(count2_values)
        selected = sorted(enumerate(objs), key=lambda kv: (-kv[1]["count2"], kv[1]["bbox"][0], kv[1]["bbox"][1]))[0]
        selected_rank, selected_obj = selected
        selected_shapes[(selected_obj["h"], selected_obj["w"])] += 1
        max_count2_hist[max_count2] += 1
        all_dense = int(all(o["nz"] == o["area"] for o in objs))
        dense_ok += all_dense
        if not np.array_equal(selected_obj["patch"], y):
            raise RuntimeError(f"rule mismatch at {idx}")
        rows.append(
            InventoryRow(
                idx=idx,
                split=split,
                object_count=len(objs),
                selected_rank=int(selected_rank),
                selected_bbox=json.dumps(selected_obj["bbox"]),
                selected_shape=f'{selected_obj["h"]}x{selected_obj["w"]}',
                max_count2=max_count2,
                count2_hist=json.dumps(dict(Counter(count2_values)), ensure_ascii=False),
                all_objects_dense_rectangles=all_dense,
                input_shape=str(tuple(x.shape)),
                output_shape=str(tuple(y.shape)),
            )
        )

    # Rough lower-bound proxy: if we can detect rectangles via color-2 counts without
    # component flood-fill, the compiler needs at least a small number of candidate
    # rectangle windows. Count the distinct selected shapes as a branch budget.
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "inventory_ready",
        "hypothesis": "task365 full-pass object crop rule may be lowerable with small rectangle/window selection because all objects are dense rectangles and input is 10x10.",
        "examples": len(examples),
        "object_count_hist": dict(object_counts),
        "selected_shape_count": len(selected_shapes),
        "selected_shape_hist": {str(k): v for k, v in selected_shapes.most_common()},
        "max_count2_hist": dict(max_count2_hist),
        "dense_rectangle_examples": dense_ok,
        "max_objects": max(object_counts),
        "decision": "If all objects are dense rectangles and shape branch count is modest, next try rectangle-window selector ONNX; otherwise avoid dynamic component lowering.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: inventory only",
        "leakage_risk": "low: structural inventory over all arc-gen, no raw lookup emitted.",
        "overfitting_risk": "medium: selected shape branches must be compressed before hidden-safe submission.",
    }
    with (EXP_DIR / "lowering_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(InventoryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task365` の `max_count2 object crop` ruleを600級ONNXへ落とせるか、矩形性とbranch数を棚卸しする。

## 結果

- examples: `{len(examples)}`
- object_count_hist: `{dict(object_counts)}`
- selected_shape_count: `{len(selected_shapes)}`
- dense_rectangle_examples: `{dense_ok}/{len(examples)}`
- max_objects: `{max(object_counts)}`

## 判断

この実験はinventoryのみでONNXは生成していない。
全objectがdense rectangleなら、dynamic componentではなくrectangle-window selectorとしてloweringを試す。

## Risk

- leakage risk: low。raw lookupなし。
- overfitting risk: medium。shape branchを説明可能に圧縮する必要あり。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

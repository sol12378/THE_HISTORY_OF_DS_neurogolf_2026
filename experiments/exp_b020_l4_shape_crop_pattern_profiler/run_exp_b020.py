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


EXP_ID = "exp_b020_l4_shape_crop_pattern_profiler"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY_PATH = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"
FAST_TOP_K = 10


@dataclass(frozen=True)
class TaskProfile:
    task_id: int
    strict_cost: int
    gain_to_250: float
    train_examples: int
    total_examples: int
    input_shape_count: int
    output_shape_count: int
    train_crop_any_pass: int
    train_anchor_rule: str
    train_anchor_pass: int
    train_anchor_total: int
    full_anchor_pass: int
    full_anchor_total: int
    bbox_crop_pass: int
    bbox_crop_total: int
    shape_to_anchor_table: str
    decision: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def load_targets() -> dict[int, dict[str, str]]:
    with TAXONOMY_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    lane_rows = [r for r in rows if r["compiler_lane"] == "L4_shape_crop_resize"]
    lane_rows = sorted(lane_rows, key=lambda r: (-float(r["gain_to_250"]), int(r["task_id"])))
    return {int(r["task_id"]): r for r in lane_rows[:FAST_TOP_K]}


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def anchor_crop(x: np.ndarray, oh: int, ow: int, anchor: str) -> np.ndarray | None:
    ih, iw = x.shape
    if oh > ih or ow > iw:
        return None
    if anchor == "tl":
        r0, c0 = 0, 0
    elif anchor == "tr":
        r0, c0 = 0, iw - ow
    elif anchor == "bl":
        r0, c0 = ih - oh, 0
    elif anchor == "br":
        r0, c0 = ih - oh, iw - ow
    elif anchor == "center":
        r0, c0 = (ih - oh) // 2, (iw - ow) // 2
    else:
        return None
    return x[r0 : r0 + oh, c0 : c0 + ow]


def any_crop_anchor(x: np.ndarray, y: np.ndarray) -> list[str]:
    if y.shape[0] > x.shape[0] or y.shape[1] > x.shape[1]:
        return []
    out = []
    for anchor in ["tl", "tr", "bl", "br", "center"]:
        crop = anchor_crop(x, y.shape[0], y.shape[1], anchor)
        if crop is not None and np.array_equal(crop, y):
            out.append(anchor)
    return out


def bbox_crop_match(x: np.ndarray, y: np.ndarray) -> bool:
    b = bbox(x != 0)
    if b is None:
        return False
    r0, c0, r1, c1 = b
    crop = x[r0:r1, c0:c1]
    return np.array_equal(crop, y)


def infer_shape_anchor_table(train: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], str]:
    table: dict[tuple[int, int, int, int], str] = {}
    for ex in train:
        x = arr(ex["input"])
        y = arr(ex["output"])
        key = (x.shape[0], x.shape[1], y.shape[0], y.shape[1])
        anchors = any_crop_anchor(x, y)
        if not anchors:
            return {}
        anchor = anchors[0]
        if key in table and table[key] != anchor:
            return {}
        table[key] = anchor
    return table


def apply_table(x: np.ndarray, oh: int, ow: int, table: dict[tuple[int, int, int, int], str]) -> np.ndarray | None:
    key = (x.shape[0], x.shape[1], oh, ow)
    anchor = table.get(key)
    if anchor is None:
        return None
    return anchor_crop(x, oh, ow, anchor)


def profile_task(task_id: int, meta: dict[str, str]) -> TaskProfile:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    input_shapes = Counter()
    output_shapes = Counter()
    train_crop_any = 0
    for ex in task["train"]:
        x = arr(ex["input"])
        y = arr(ex["output"])
        if any_crop_anchor(x, y):
            train_crop_any += 1
    table = infer_shape_anchor_table(list(task["train"]))
    train_anchor_pass = 0
    full_anchor_pass = 0
    bbox_pass = 0
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        input_shapes[x.shape] += 1
        output_shapes[y.shape] += 1
        pred = apply_table(x, y.shape[0], y.shape[1], table) if table else None
        if pred is not None and np.array_equal(pred, y):
            full_anchor_pass += 1
            if idx < len(task["train"]):
                train_anchor_pass += 1
        if bbox_crop_match(x, y):
            bbox_pass += 1
    if table and full_anchor_pass == len(examples):
        decision = "lower_candidate: static shape-anchor Slice table"
    elif table and train_anchor_pass == len(task["train"]):
        decision = "train_only_shape_anchor: needs generalization/holdout"
    elif train_crop_any:
        decision = "partial_anchor_crop_signal"
    else:
        decision = "not_simple_crop_anchor"
    return TaskProfile(
        task_id=task_id,
        strict_cost=int(float(meta["strict_cost"])),
        gain_to_250=float(meta["gain_to_250"]),
        train_examples=len(task["train"]),
        total_examples=len(examples),
        input_shape_count=len(input_shapes),
        output_shape_count=len(output_shapes),
        train_crop_any_pass=train_crop_any,
        train_anchor_rule=json.dumps({str(k): v for k, v in table.items()}, ensure_ascii=False),
        train_anchor_pass=train_anchor_pass,
        train_anchor_total=len(task["train"]),
        full_anchor_pass=full_anchor_pass,
        full_anchor_total=len(examples),
        bbox_crop_pass=bbox_pass,
        bbox_crop_total=len(examples),
        shape_to_anchor_table=json.dumps({str(k): v for k, v in table.items()}, ensure_ascii=False),
        decision=decision,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    rows = [profile_task(task_id, meta) for task_id, meta in targets.items()]
    rows = sorted(rows, key=lambda r: (r.decision != "lower_candidate: static shape-anchor Slice table", -r.full_anchor_pass, -r.gain_to_250, r.task_id))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "lower_candidate_found" if any(r.decision.startswith("lower_candidate") for r in rows) else "profile_ready",
        "target_task_count": len(targets),
        "fast_top_k": FAST_TOP_K,
        "target_task_ids": sorted(targets),
        "lower_candidate_count": sum(1 for r in rows if r.decision.startswith("lower_candidate")),
        "profiles": [asdict(r) for r in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: profile only; lower candidates first",
        "decision": "lower static shape-anchor candidates if any; otherwise inspect object-anchor crop features",
        "leakage_risk": "low-to-medium: train-inferred shape table, evaluated on all arc-gen.",
        "overfitting_risk": "medium unless table full-passes all arc-gen and branch count is tiny.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "task_profiles.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

L4 shape/crop lane top taskについて、固定anchor crop / shape-anchor table / nonzero bbox cropで説明できるかをprofileする。

## 結果

- target tasks: {len(targets)}
- lower candidates: {result["lower_candidate_count"]}
- top profiles: {[asdict(r) for r in rows[:5]]}

## 判断

full passのstatic shape-anchor候補があればSlice loweringへ進む。なければobject-anchor crop profileへ進む。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

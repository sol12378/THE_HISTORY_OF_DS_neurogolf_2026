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


EXP_ID = "exp065_task020_lowering_reference"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


GROUPS = {
    "corners": {(0, 0), (0, 4), (4, 0), (4, 4)},
    "edge_mid": {(0, 2), (2, 0), (2, 4), (4, 2)},
    "inner": {(1, 1), (1, 3), (3, 1), (3, 3)},
    "center": {(2, 2)},
}

TEMPLATES = {
    "corners": ((0, 0), (0, 4), (4, 0), (4, 4)),
    "edge_mid": ((0, 2), (2, 0), (2, 4), (4, 2)),
    "inner_diag": ((1, 1), (1, 3), (3, 1), (3, 3)),
}


@dataclass(frozen=True)
class EvalRow:
    example_idx: int
    split: str
    status: str
    bbox: str
    target_color: int
    class_name: str
    variant: str
    fill_cells: str
    reason: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def d4_variants(points: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    h = w = 5
    variants = []
    for mode in range(8):
        out = []
        for r, c in points:
            if mode == 0:
                rr, cc = r, c
            elif mode == 1:
                rr, cc = r, w - 1 - c
            elif mode == 2:
                rr, cc = h - 1 - r, c
            elif mode == 3:
                rr, cc = h - 1 - r, w - 1 - c
            elif mode == 4:
                rr, cc = c, r
            elif mode == 5:
                rr, cc = c, h - 1 - r
            elif mode == 6:
                rr, cc = w - 1 - c, r
            else:
                rr, cc = w - 1 - c, h - 1 - r
            out.append((rr, cc))
        variants.append(tuple(sorted(out)))
    return sorted(set(variants))


def choose_target_color(local: np.ndarray) -> int | None:
    best: tuple[int, int] | None = None
    for color in [int(v) for v in np.unique(local) if int(v) != 0]:
        pos = {(int(r), int(c)) for r, c in np.argwhere(local == color)}
        if not pos:
            continue
        counts = {
            name: len(pos & group)
            for name, group in GROUPS.items()
        }
        if counts["center"] == len(pos):
            continue
        # exp062 selector is meaningful when any orbit group has exactly one member
        # or when inner is present. Choose the most constrained color.
        score = 0
        score += 10 if counts["corners"] == 1 else 0
        score += 10 if counts["edge_mid"] == 1 else 0
        score += 5 if counts["inner"] > 0 else 0
        score -= len(pos)
        cand = (score, color)
        if best is None or cand > best:
            best = cand
    if best is None or best[0] <= -99:
        return None
    return best[1]


def class_from_counts(local: np.ndarray, color: int) -> str:
    pos = {(int(r), int(c)) for r, c in np.argwhere(local == color)}
    corners_count = len(pos & GROUPS["corners"])
    edge_mid_count = len(pos & GROUPS["edge_mid"])
    inner_count = len(pos & GROUPS["inner"])
    if corners_count == 1:
        return "corners"
    if edge_mid_count == 1:
        return "edge_mid"
    if inner_count > 0:
        return "inner_diag"
    return "corners"


def apply_reference(x: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    b = bbox(x != 0)
    if b is None:
        return x.copy(), {"reason": "empty"}
    r0, c0, r1, c1 = b
    if (r1 - r0, c1 - c0) != (5, 5):
        return x.copy(), {"reason": f"bbox_shape={(r1-r0, c1-c0)}", "bbox": b}
    local = x[r0:r1, c0:c1]
    color = choose_target_color(local)
    if color is None:
        return x.copy(), {"reason": "no_target_color", "bbox": b}
    cls = class_from_counts(local, color)
    color_pos = {(int(r), int(c)) for r, c in np.argwhere(local == color)}
    candidates = []
    for variant in d4_variants(TEMPLATES[cls]):
        variant_set = set(variant)
        missing = variant_set - color_pos
        if len(missing) != 3:
            continue
        if all(local[rr, cc] == 0 for rr, cc in missing):
            candidates.append((variant, missing))
    if len(candidates) != 1:
        return x.copy(), {"reason": f"candidate_count={len(candidates)}", "bbox": b, "target_color": color, "class_name": cls}
    variant, missing = candidates[0]
    out = x.copy()
    for rr, cc in missing:
        out[r0 + rr, c0 + cc] = color
    return out, {"reason": "ok", "bbox": b, "target_color": color, "class_name": cls, "variant": variant, "fill_cells": tuple(sorted(missing))}


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    rows: list[EvalRow] = []
    split_counts = Counter()
    split_pass = Counter()
    reasons = Counter()
    class_counts = Counter()
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred, info = apply_reference(x)
        ok = bool(np.array_equal(pred, y))
        split_counts[split] += 1
        if ok:
            split_pass[split] += 1
        reasons[info.get("reason", "")] += 1
        class_counts[info.get("class_name", "")] += 1
        rows.append(
            EvalRow(
                example_idx=idx,
                split=split,
                status="pass" if ok else "fail",
                bbox=str(info.get("bbox", "")),
                target_color=int(info.get("target_color", -1)),
                class_name=str(info.get("class_name", "")),
                variant=str(info.get("variant", "")),
                fill_cells=str(info.get("fill_cells", "")),
                reason=str(info.get("reason", "")),
            )
        )
    pass_count = sum(1 for r in rows if r.status == "pass")
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "reference_pass" if pass_count == len(rows) else "reference_partial",
        "target_task": TARGET_TASK,
        "total_pass": pass_count,
        "total_examples": len(rows),
        "split_counts": dict(split_counts),
        "split_pass": dict(split_pass),
        "reason_counts": dict(reasons),
        "class_counts": dict(class_counts),
        "decision": "Use this reference as the block-level oracle for ONNX lowering. If partial, fix target-color selection before ONNX.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: reference only",
        "leakage_risk": "low: rule is explicit and nonlookup.",
        "overfitting_risk": "medium: task-specific; still needs ONNX and Kaggle delta after cost improvement.",
        "outputs": {"eval_rows": "eval_rows.csv", "notes": "notes.md"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "eval_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

task020 exp062 ruleをONNX化する前に、bbox crop / target color / class selector / orientation / writebackのreference実装を固定する。

## 結果

- pass: {pass_count}/{len(rows)}
- split pass: {dict(split_pass)} / {dict(split_counts)}
- reasons: {dict(reasons)}
- class counts: {dict(class_counts)}

## Decision

referenceがfull passならONNX化へ進む。partialならtarget-color selectionやclass selectorを修正する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

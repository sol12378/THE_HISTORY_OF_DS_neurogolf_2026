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
from experiments.exp080_task366_marker_object_copy_rule.run_exp080 import (  # noqa: E402
    TASK_ID,
    arr,
    background,
    bbox,
    components,
    rel_shape,
    split_source_target,
)


EXP_ID = "exp081_task366_lowering_inventory"
EXP_DIR = ROOT / "experiments" / EXP_ID


@dataclass(frozen=True)
class TemplateRow:
    template_id: str
    count: int
    patch_shape: str
    patch_cells: int
    marker_color_count: int
    marker_signature: str
    estimated_kernel_params: int
    lowering_role: str


@dataclass(frozen=True)
class ExampleLoweringStats:
    idx: int
    split: str
    input_shape: str
    output_shape: str
    source_objects: int
    used_objects: int
    distractor_objects: int
    marker_cells: int
    max_patch_cells: int
    unique_templates: int


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def patch_signature(patch: np.ndarray, src_bg: int) -> tuple[str, int, int, int]:
    h, w = patch.shape
    nz = [(int(r), int(c), int(patch[r, c])) for r, c in np.argwhere(patch != src_bg)]
    marker_parts = []
    for color in [int(v) for v in np.unique(patch) if int(v) != src_bg]:
        coords = [(int(r), int(c)) for r, c in np.argwhere(patch == color)]
        marker_parts.append((color, tuple(sorted(rel_shape(coords)))))
    sig = json.dumps(
        {
            "shape": [h, w],
            "nz": sorted(nz),
            "markers": marker_parts,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    marker_color_count = len(marker_parts)
    return sig, h, w, marker_color_count


def source_templates_for_example(x: np.ndarray, out_shape: tuple[int, int]) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray] | None:
    panels = split_source_target(x, out_shape)
    if panels is None:
        return None
    source, target = panels
    src_bg = background(source)
    records: list[dict[str, Any]] = []
    for comp in components(source != src_bg):
        r0, c0, r1, c1 = bbox(comp)
        patch = source[r0:r1, c0:c1]
        sig, h, w, marker_color_count = patch_signature(patch, src_bg)
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "signature": sig,
                "h": h,
                "w": w,
                "marker_color_count": marker_color_count,
                "patch_cells": int(np.count_nonzero(patch != src_bg)),
            }
        )
    return records, source, target


def marker_coverage_needed(target: np.ndarray) -> set[tuple[int, int, int]]:
    tgt_bg = background(target)
    return {
        (int(r), int(c), int(target[r, c]))
        for r, c in np.argwhere(target != tgt_bg)
    }


def count_used_signatures(records: list[dict[str, Any]], target: np.ndarray) -> set[str]:
    # Reuse the rule's core idea for inventory: a template is potentially used if any
    # non-background color in its patch appears in the marker panel.
    target_colors = {int(v) for v in np.unique(target) if int(v) != background(target)}
    used = set()
    for rec in records:
        colors = {int(v) for v in np.unique(rec["patch"]) if int(v) != background(rec["patch"])}
        if colors & target_colors:
            used.add(rec["signature"])
    return used


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    template_counts: Counter[str] = Counter()
    template_meta: dict[str, dict[str, Any]] = {}
    example_rows: list[ExampleLoweringStats] = []
    used_template_counts: Counter[str] = Counter()
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        payload = source_templates_for_example(x, y.shape)
        if payload is None:
            continue
        records, _, target = payload
        used = count_used_signatures(records, target)
        for rec in records:
            sig = rec["signature"]
            template_counts[sig] += 1
            template_meta.setdefault(sig, rec)
            if sig in used:
                used_template_counts[sig] += 1
        example_rows.append(
            ExampleLoweringStats(
                idx=idx,
                split=split,
                input_shape=str(tuple(x.shape)),
                output_shape=str(tuple(y.shape)),
                source_objects=len(records),
                used_objects=len(used),
                distractor_objects=max(0, len(records) - len(used)),
                marker_cells=len(marker_coverage_needed(target)),
                max_patch_cells=max((rec["patch_cells"] for rec in records), default=0),
                unique_templates=len({rec["signature"] for rec in records}),
            )
        )

    template_rows: list[TemplateRow] = []
    for i, (sig, count) in enumerate(template_counts.most_common()):
        rec = template_meta[sig]
        h, w = int(rec["h"]), int(rec["w"])
        patch_cells = int(rec["patch_cells"])
        marker_color_count = int(rec["marker_color_count"])
        # A direct Conv template detector would need one channel per non-bg color and
        # roughly h*w*colors parameters. This is a deliberately conservative proxy.
        colors = {int(v) for v in np.unique(rec["patch"]) if int(v) != background(rec["patch"])}
        estimated_kernel_params = h * w * max(1, len(colors))
        role = "common_used_template" if used_template_counts[sig] else "distractor_or_rare"
        template_rows.append(
            TemplateRow(
                template_id=f"T{i:03d}",
                count=count,
                patch_shape=f"{h}x{w}",
                patch_cells=patch_cells,
                marker_color_count=marker_color_count,
                marker_signature=sig[:240],
                estimated_kernel_params=estimated_kernel_params,
                lowering_role=role,
            )
        )

    total_kernel_params = sum(r.estimated_kernel_params for r in template_rows if r.lowering_role == "common_used_template")
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "inventory_ready",
        "hypothesis": "task366 full-pass rule can be lowered if source object templates and marker placements are few enough for static Conv/Slice/Where templates.",
        "examples": len(examples),
        "unique_source_templates": len(template_rows),
        "common_used_templates": sum(1 for r in template_rows if r.lowering_role == "common_used_template"),
        "max_source_objects_per_example": max(r.source_objects for r in example_rows),
        "max_used_objects_per_example": max(r.used_objects for r in example_rows),
        "max_marker_cells": max(r.marker_cells for r in example_rows),
        "max_patch_cells": max(r.max_patch_cells for r in example_rows),
        "estimated_total_used_kernel_params": total_kernel_params,
        "decision": "If used template count and kernel proxy are modest, next emit a correctness-first static template detector; otherwise continue rule compression before ONNX.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: inventory only; no ONNX emitted",
        "leakage_risk": "low-to-medium: uses all arc-gen to size compiler templates, but does not emit raw lookup.",
        "overfitting_risk": "medium: template inventory must be compressed to structural rules before hidden-safe submission.",
    }

    with (EXP_DIR / "template_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TemplateRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in template_rows])
    with (EXP_DIR / "example_lowering_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleLoweringStats.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in example_rows])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp080` のtask366 full-pass ruleをONNX化する前に、static template detectorで小さく表現できるか棚卸しする。

## 結果

- examples: `{len(examples)}`
- unique source templates: `{len(template_rows)}`
- common used templates: `{result["common_used_templates"]}`
- max source objects/example: `{result["max_source_objects_per_example"]}`
- max used objects/example: `{result["max_used_objects_per_example"]}`
- max marker cells: `{result["max_marker_cells"]}`
- estimated total used kernel params: `{total_kernel_params}`

## 判断

この実験はinventoryのみで、ONNXは生成していない。
template数とkernel proxyが小さければ、次はcorrectness-first static detectorを生成する。
大きすぎる場合は、object shapeを個別templateではなく構造ruleへ圧縮する。

## Risk

- leakage risk: low-to-medium。all arc-genでcompiler sizeを測るが、raw lookup提出物は生成しない。
- overfitting risk: medium。template列挙のままではhidden汎化が弱い可能性がある。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

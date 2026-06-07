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
from experiments.exp058_task020_residual_template_audit.run_exp058 import canonical_template  # noqa: E402
from experiments.exp059_task020_three_template_selector.run_exp059 import TEMPLATES, d4_variants  # noqa: E402


EXP_ID = "exp060_task020_orientation_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class ExampleRow:
    example_idx: int
    split: str
    target_color: int
    template_name: str
    bbox_shape: str
    input_color_pos: str
    true_missing: str
    true_full_variant: str
    compatible_variant_count: int
    compatible_variants: str
    ambiguity_kind: str


@dataclass(frozen=True)
class SelectorRow:
    selector_name: str
    template_accuracy: int
    variant_accuracy: int
    output_pass: int
    total_examples: int
    fail_reasons: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def rel_positions(mask: np.ndarray, b: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = b
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in np.argwhere(mask)))


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def true_info(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    b = bbox(x != 0)
    assert b is not None
    shape = (b[2] - b[0], b[3] - b[1])
    changed = x != y
    colors = [int(v) for v in np.unique(y[changed]) if int(v) != 0]
    color = colors[0]
    missing = rel_positions(changed & (y == color), b)
    canon = canonical_template(missing, shape)
    template_name = next(name for name, template in TEMPLATES.items() if tuple(template) == tuple(canon))
    input_pos = rel_positions(x[b[0] : b[2], b[1] : b[3]] == color, (0, 0, shape[0], shape[1]))
    true_full = tuple(sorted(set(input_pos) | set(missing)))
    compatible = []
    for variant in d4_variants(TEMPLATES[template_name], shape):
        variant_set = set(variant)
        miss = variant_set - set(input_pos)
        if len(miss) == len(missing) and set(missing) == miss:
            compatible.append(variant)
    any_compatible = []
    for variant in d4_variants(TEMPLATES[template_name], shape):
        variant_set = set(variant)
        miss = variant_set - set(input_pos)
        if len(miss) == len(missing):
            any_compatible.append(variant)
    return {
        "bbox": b,
        "shape": shape,
        "target_color": color,
        "missing": missing,
        "template_name": template_name,
        "input_pos": input_pos,
        "true_full": true_full,
        "compatible": compatible,
        "any_compatible": any_compatible,
    }


def predict_by_selector(selector: str, info: dict[str, Any]) -> tuple[str | None, tuple[tuple[int, int], ...] | None]:
    input_pos = set(info["input_pos"])
    shape = info["shape"]
    # For now selectors assume the true class is known; this isolates orientation.
    template_name = info["template_name"]
    variants = d4_variants(TEMPLATES[template_name], shape)
    candidates = []
    for variant in variants:
        miss = set(variant) - input_pos
        if len(miss) == len(info["missing"]):
            candidates.append(variant)
    if not candidates:
        return template_name, None
    if selector == "first_lexicographic":
        return template_name, candidates[0]
    if selector == "min_missing_lexicographic":
        return template_name, sorted(candidates, key=lambda v: tuple(sorted(set(v) - input_pos)))[0]
    if selector == "prefer_variant_containing_input":
        containing = [v for v in candidates if input_pos <= set(v)]
        return template_name, (containing[0] if containing else candidates[0])
    if selector == "oracle_if_unique":
        return template_name, (candidates[0] if len(candidates) == 1 else None)
    raise ValueError(selector)


def apply_variant(x: np.ndarray, info: dict[str, Any], variant: tuple[tuple[int, int], ...] | None) -> np.ndarray:
    if variant is None:
        return x.copy()
    out = x.copy()
    b = info["bbox"]
    color = info["target_color"]
    input_pos = set(info["input_pos"])
    missing = set(variant) - input_pos
    if len(missing) != len(info["missing"]):
        return out
    for rr, cc in missing:
        if out[b[0] + rr, b[1] + cc] != 0:
            return x.copy()
    for rr, cc in missing:
        out[b[0] + rr, b[1] + cc] = color
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])

    example_rows: list[ExampleRow] = []
    infos = []
    ambiguity = Counter()
    compat_counts = Counter()
    full_variant_counts = Counter()
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        info = true_info(x, y)
        infos.append((idx, x, y, info))
        compat_count = len(info["any_compatible"])
        compat_counts[compat_count] += 1
        full_variant_counts[str(info["true_full"])] += 1
        if compat_count == 0:
            kind = "no_candidate"
        elif compat_count == 1:
            kind = "unique_orientation"
        else:
            kind = "ambiguous_orientation"
        ambiguity[kind] += 1
        example_rows.append(
            ExampleRow(
                example_idx=idx,
                split=split_name(idx, train_n, test_n),
                target_color=info["target_color"],
                template_name=info["template_name"],
                bbox_shape=str(info["shape"]),
                input_color_pos=str(info["input_pos"]),
                true_missing=str(info["missing"]),
                true_full_variant=str(info["true_full"]),
                compatible_variant_count=compat_count,
                compatible_variants=str(info["any_compatible"][:8]),
                ambiguity_kind=kind,
            )
        )

    selector_rows: list[SelectorRow] = []
    for selector in ["first_lexicographic", "min_missing_lexicographic", "prefer_variant_containing_input", "oracle_if_unique"]:
        variant_ok = 0
        output_ok = 0
        fail = Counter()
        for idx, x, y, info in infos:
            _, variant = predict_by_selector(selector, info)
            if variant is None:
                fail["no_variant"] += 1
                pred = x
            else:
                if tuple(variant) == tuple(info["true_full"]):
                    variant_ok += 1
                else:
                    fail["wrong_variant"] += 1
                pred = apply_variant(x, info, variant)
            if np.array_equal(pred, y):
                output_ok += 1
            else:
                fail["output_mismatch"] += 1
        selector_rows.append(
            SelectorRow(
                selector_name=selector,
                template_accuracy=len(infos),
                variant_accuracy=variant_ok,
                output_pass=output_ok,
                total_examples=len(infos),
                fail_reasons=json.dumps(dict(fail.most_common()), ensure_ascii=False),
            )
        )

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_ready",
        "target_task": TARGET_TASK,
        "example_count": len(infos),
        "ambiguity_counts": dict(ambiguity),
        "compatible_variant_count_histogram": dict(compat_counts),
        "unique_true_full_variants": len(full_variant_counts),
        "top_true_full_variants": dict(full_variant_counts.most_common(20)),
        "selector_results": [asdict(r) for r in selector_rows],
        "decision": "If most examples are ambiguous, add orientation features from input color positions; if unique, fix apply logic.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic audit only",
        "leakage_risk": "medium: all examples used for diagnosis; do not emit full variant table.",
        "overfitting_risk": "medium-high if true_full variants are used as lookup.",
        "outputs": {
            "example_orientation": "example_orientation.csv",
            "selector_orientation_eval": "selector_orientation_eval.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "example_orientation.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in example_rows])
    with (EXP_DIR / "selector_orientation_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SelectorRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in selector_rows])
    notes = f"""# {EXP_ID}

## 目的

task020の3-template selectorで残るD4 orientation問題を分離して診断する。

## 結果

- examples: {len(infos)}
- ambiguity counts: {dict(ambiguity)}
- compatible variant count histogram: {dict(compat_counts)}
- unique true full variants: {len(full_variant_counts)}

## Selector Results

| selector | variant acc | output pass |
|---|---:|---:|
"""
    for row in selector_rows:
        notes += f"| {row.selector_name} | {row.variant_accuracy}/{row.total_examples} | {row.output_pass}/{row.total_examples} |\n"
    notes += """
## Decision

orientationが曖昧なら、入力色位置からorientation featureを追加する。true full variant tableをそのまま提出してはいけない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

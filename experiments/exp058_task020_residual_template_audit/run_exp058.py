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
from experiments.exp052_task020_d4_orbit_rule.run_exp052 import apply_rule as d4_apply
from experiments.exp052_task020_d4_orbit_rule.run_exp052 import orbit_library


EXP_ID = "exp058_task020_residual_template_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class TemplateRow:
    template_rank: int
    template: str
    count: int
    d4_pass_count: int
    d4_fail_count: int
    colors: str
    example_indices: str


@dataclass(frozen=True)
class ExampleRow:
    example_idx: int
    split: str
    d4_status: str
    target_color: int
    bbox_shape: str
    rel_template: str
    input_color_positions: str
    missing_from_d4_reason: str


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


def rel_positions(mask: np.ndarray, b: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = b
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in np.argwhere(mask)))


def canonical_template(points: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    h, w = shape
    transforms = []
    for r, c in points:
        transforms.append(
            [
                (r, c),
                (r, w - 1 - c),
                (h - 1 - r, c),
                (h - 1 - r, w - 1 - c),
                (c, r) if h == w else (r, c),
                (c, h - 1 - r) if h == w else (r, c),
                (w - 1 - c, r) if h == w else (r, c),
                (w - 1 - c, h - 1 - r) if h == w else (r, c),
            ]
        )
    variants = []
    for i in range(8):
        variants.append(tuple(sorted(t[i] for t in transforms)))
    return min(variants)


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    orbits = orbit_library()
    example_rows: list[ExampleRow] = []
    template_counter: Counter[str] = Counter()
    template_d4_pass: Counter[str] = Counter()
    template_d4_fail: Counter[str] = Counter()
    template_colors: dict[str, Counter[int]] = {}
    template_examples: dict[str, list[int]] = {}

    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        b = bbox(x != 0)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        shape = (r1 - r0, c1 - c0)
        changed = x != y
        colors = [int(v) for v in np.unique(y[changed]) if int(v) != 0]
        target_color = colors[0] if colors else -1
        rel = rel_positions(changed & (y == target_color), b)
        canon = canonical_template(rel, shape)
        template = str(canon)
        pred, _, _, reason = d4_apply(x, orbits)
        d4_ok = bool(np.array_equal(pred, y))
        template_counter[template] += 1
        if d4_ok:
            template_d4_pass[template] += 1
        else:
            template_d4_fail[template] += 1
        template_colors.setdefault(template, Counter())[target_color] += 1
        template_examples.setdefault(template, []).append(idx)
        color_pos = rel_positions(x == target_color, b)
        example_rows.append(
            ExampleRow(
                example_idx=idx,
                split=split,
                d4_status="pass" if d4_ok else "fail",
                target_color=target_color,
                bbox_shape=str(shape),
                rel_template=str(rel),
                input_color_positions=str(color_pos),
                missing_from_d4_reason=reason,
            )
        )

    template_rows = []
    for rank, (template, count) in enumerate(template_counter.most_common(), start=1):
        template_rows.append(
            TemplateRow(
                template_rank=rank,
                template=template,
                count=count,
                d4_pass_count=template_d4_pass[template],
                d4_fail_count=template_d4_fail[template],
                colors=json.dumps(dict(template_colors[template].most_common()), ensure_ascii=False),
                example_indices=" ".join(str(i) for i in template_examples[template][:30]),
            )
        )

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "audit_ready",
        "target_task": TARGET_TASK,
        "example_count": len(example_rows),
        "unique_canonical_templates": len(template_rows),
        "top_templates": [asdict(r) for r in template_rows[:20]],
        "d4_fail_template_count": sum(1 for r in template_rows if r.d4_fail_count > 0),
        "decision": "Use residual templates to design the next task020 object-role/local-frame rule; do not lower until a full-pass rule exists.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic audit only",
        "leakage_risk": "medium: all examples are used to inspect residuals; use only to design low-complexity rules, not label tables.",
        "overfitting_risk": "medium-high: residual templates can become memorization if emitted directly.",
        "outputs": {
            "template_summary": "template_summary.csv",
            "example_templates": "example_templates.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "template_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TemplateRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in template_rows])
    with (EXP_DIR / "example_templates.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in example_rows])
    notes = f"""# {EXP_ID}

## 目的

task020でD4 orbit ruleが漏らす106例について、真の変更セルtemplateをbbox-local座標で分類する。

## 結果

- examples: {len(example_rows)}
- unique canonical templates: {len(template_rows)}
- d4 fail template count: {result['d4_fail_template_count']}

## Top Templates

| rank | count | d4 pass | d4 fail | template |
|---:|---:|---:|---:|---|
"""
    for row in template_rows[:12]:
        notes += f"| {row.template_rank} | {row.count} | {row.d4_pass_count} | {row.d4_fail_count} | `{row.template}` |\n"
    notes += """
## Decision

この結果はrule設計用の診断であり、template tableとしてONNXへ出してはいけない。次は少数の幾何templateをobject-role条件で選択できるかを調べる。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

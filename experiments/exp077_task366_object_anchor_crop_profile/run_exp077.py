from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp077_task366_object_anchor_crop_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 366


@dataclass(frozen=True)
class ExampleProfile:
    idx: int
    split: str
    input_shape: str
    output_shape: str
    exact_crop_matches: int
    exact_crop_positions: str
    nonzero_bbox_match: int
    color_bbox_matches: str
    component_bbox_matches: int
    component_bbox_shapes: str
    crop_with_color_remap_matches: int
    best_signal: str


@dataclass(frozen=True)
class RuleEval:
    candidate: str
    status: str
    train_pass: int
    train_total: int
    test_pass: int
    test_total: int
    arc_pass: int
    arc_total: int
    total_pass: int
    total: int
    fail_examples: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def crop(x: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return x[r0:r1, c0:c1]


def exact_crop_positions(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int]]:
    if y.shape[0] > x.shape[0] or y.shape[1] > x.shape[1]:
        return []
    out: list[tuple[int, int]] = []
    oh, ow = y.shape
    for r in range(x.shape[0] - oh + 1):
        for c in range(x.shape[1] - ow + 1):
            if np.array_equal(x[r : r + oh, c : c + ow], y):
                out.append((r, c))
    return out


def infer_color_remap(src: np.ndarray, dst: np.ndarray) -> dict[int, int] | None:
    if src.shape != dst.shape:
        return None
    mapping: dict[int, int] = {}
    for a, b in zip(src.ravel(), dst.ravel()):
        aa, bb = int(a), int(b)
        if aa in mapping and mapping[aa] != bb:
            return None
        mapping[aa] = bb
    return mapping


def remap_crop_positions(x: np.ndarray, y: np.ndarray) -> list[tuple[int, int, dict[int, int]]]:
    if y.shape[0] > x.shape[0] or y.shape[1] > x.shape[1]:
        return []
    out: list[tuple[int, int, dict[int, int]]] = []
    oh, ow = y.shape
    for r in range(x.shape[0] - oh + 1):
        for c in range(x.shape[1] - ow + 1):
            mapping = infer_color_remap(x[r : r + oh, c : c + ow], y)
            if mapping is not None:
                out.append((r, c, mapping))
    return out


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def anchor_for_position(x_shape: tuple[int, int], y_shape: tuple[int, int], pos: tuple[int, int]) -> str:
    r, c = pos
    ih, iw = x_shape
    oh, ow = y_shape
    anchors = []
    if r == 0:
        anchors.append("top")
    if r + oh == ih:
        anchors.append("bottom")
    if c == 0:
        anchors.append("left")
    if c + ow == iw:
        anchors.append("right")
    if r == (ih - oh) // 2 and c == (iw - ow) // 2:
        anchors.append("center")
    return "+".join(anchors) if anchors else f"offset({r},{c})"


def profile_example(idx: int, split: str, x: np.ndarray, y: np.ndarray) -> ExampleProfile:
    positions = exact_crop_positions(x, y)
    nonzero_box = bbox(x != 0)
    nonzero_match = int(nonzero_box is not None and np.array_equal(crop(x, nonzero_box), y))
    color_matches: list[dict[str, Any]] = []
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        b = bbox(x == color)
        if b is not None and np.array_equal(crop(x, b), y):
            color_matches.append({"color": color, "bbox": b})
    comp_matches = 0
    comp_shapes = Counter()
    for comp in components(x != 0):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        b = (min(rs), min(cs), max(rs) + 1, max(cs) + 1)
        comp_shapes[(b[2] - b[0], b[3] - b[1])] += 1
        if np.array_equal(crop(x, b), y):
            comp_matches += 1
    remap_positions = remap_crop_positions(x, y)
    if positions:
        best = "exact_crop:" + ",".join(anchor_for_position(x.shape, y.shape, p) for p in positions[:3])
    elif nonzero_match:
        best = "nonzero_bbox"
    elif color_matches:
        best = "color_bbox"
    elif comp_matches:
        best = "component_bbox"
    elif remap_positions:
        best = "crop_with_color_remap"
    else:
        best = "no_crop_signal"
    return ExampleProfile(
        idx=idx,
        split=split,
        input_shape=str(tuple(x.shape)),
        output_shape=str(tuple(y.shape)),
        exact_crop_matches=len(positions),
        exact_crop_positions=json.dumps(positions[:20], ensure_ascii=False),
        nonzero_bbox_match=nonzero_match,
        color_bbox_matches=json.dumps(color_matches[:20], ensure_ascii=False),
        component_bbox_matches=comp_matches,
        component_bbox_shapes=json.dumps({str(k): v for k, v in comp_shapes.items()}, ensure_ascii=False),
        crop_with_color_remap_matches=len(remap_positions),
        best_signal=best,
    )


def predict_exact_train_anchor(x: np.ndarray, oh: int, ow: int, table: dict[tuple[int, int, int, int], tuple[int, int]]) -> np.ndarray | None:
    pos = table.get((x.shape[0], x.shape[1], oh, ow))
    if pos is None:
        return None
    r, c = pos
    if r + oh > x.shape[0] or c + ow > x.shape[1]:
        return None
    return x[r : r + oh, c : c + ow]


def infer_train_exact_table(task: dict[str, Any]) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    table: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for ex in task["train"]:
        x = arr(ex["input"])
        y = arr(ex["output"])
        positions = exact_crop_positions(x, y)
        if len(positions) != 1:
            return {}
        key = (x.shape[0], x.shape[1], y.shape[0], y.shape[1])
        if key in table and table[key] != positions[0]:
            return {}
        table[key] = positions[0]
    return table


def evaluate_train_exact_table(task: dict[str, Any], table: dict[tuple[int, int, int, int], tuple[int, int]]) -> RuleEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    pass_counts = Counter()
    total_counts = Counter()
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        total_counts[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = predict_exact_train_anchor(x, y.shape[0], y.shape[1], table)
        ok = pred is not None and np.array_equal(pred, y)
        if ok:
            pass_counts[split] += 1
        elif len(fails) < 20:
            fails.append({"idx": idx, "split": split, "in": tuple(x.shape), "out": tuple(y.shape)})
    total_pass = sum(pass_counts.values())
    total = sum(total_counts.values())
    return RuleEval(
        candidate="train_exact_crop_position_table",
        status="full_pass" if total_pass == total else "partial",
        train_pass=pass_counts["train"],
        train_total=total_counts["train"],
        test_pass=pass_counts["test"],
        test_total=total_counts["test"],
        arc_pass=pass_counts["arc-gen"],
        arc_total=total_counts["arc-gen"],
        total_pass=total_pass,
        total=total,
        fail_examples=json.dumps(fails, ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    profiles: list[ExampleProfile] = []
    for idx, ex in enumerate(examples):
        profiles.append(profile_example(idx, split_name(idx, train_n, test_n), arr(ex["input"]), arr(ex["output"])))
    signal_counts = Counter(p.best_signal.split(":")[0] for p in profiles)
    shape_pairs = Counter((p.input_shape, p.output_shape) for p in profiles)
    table = infer_train_exact_table(task)
    eval_row = evaluate_train_exact_table(task, table) if table else RuleEval(
        candidate="train_exact_crop_position_table",
        status="no_table",
        train_pass=0,
        train_total=train_n,
        test_pass=0,
        test_total=test_n,
        arc_pass=0,
        arc_total=len(examples) - train_n - test_n,
        total_pass=0,
        total=len(examples),
        fail_examples="[]",
    )
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "lower_candidate_found" if eval_row.status == "full_pass" else "profile_ready",
        "hypothesis": "task366は巨大なcrop/resize lane候補であり、object-anchor cropまたは色remap付きcropとして低cost Slice/Gatherへ落とせる可能性がある。",
        "examples": len(examples),
        "signal_counts": dict(signal_counts),
        "shape_pair_count": len(shape_pairs),
        "top_shape_pairs": [{"input": k[0], "output": k[1], "n": v} for k, v in shape_pairs.most_common(10)],
        "train_exact_table": {str(k): v for k, v in table.items()},
        "train_exact_table_eval": asdict(eval_row),
        "decision": "exact crop tableがfull passならstatic Slice lowering候補。そうでなければremap/component/object-anchor featureを追加する。",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: profile only; no ONNX emitted",
        "leakage_risk": "low-to-medium: train-derived crop table is evaluated on all arc-gen; do not submit unless tiny/full-pass and nonlookup.",
        "overfitting_risk": "medium: shape-position tables can overfit generated shapes; require all-arc full pass and branch compression.",
    }
    with (EXP_DIR / "example_profiles.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ExampleProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(p) for p in profiles])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

最大gain候補の `task366` について、低cost `Slice` / `Gather` へ落とせるcrop ruleがあるかを調べる。

## 結果

- examples: `{len(examples)}`
- signal counts: `{dict(signal_counts)}`
- shape pair count: `{len(shape_pairs)}`
- train exact table eval: `{eval_row.total_pass}/{eval_row.total}` (`{eval_row.status}`)

## 判断

この実験はprofileのみで、ONNXは生成していない。
exact crop / nonzero bbox / component bbox / color-remap crop の信号を見て、次のobject-anchor crop compilerを絞る。

## Risk

- leakage risk: low-to-medium。train tableは全arc-genで評価するが、raw tableをそのまま提出しない。
- overfitting risk: medium。shape-position tableはbranch圧縮が必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

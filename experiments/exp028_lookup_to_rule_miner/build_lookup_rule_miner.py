from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from collections import Counter
from datetime import date
from typing import Any, Callable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.synthesis.orchestrator import load_csv, write_csv  # noqa: E402


EXP_ID = "exp028_lookup_to_rule_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
QUEUE_PATH = ROOT / "experiments" / "exp026_synthesis_orchestrator_core" / "synthesis_queue.csv"
BASE_RESULT_PATH = ROOT / "experiments" / "exp023_graph_surgery_exp016" / "result.json"


Grid = np.ndarray
RuleFn = Callable[[Grid], Grid]


def point(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def grid(raw: list[list[int]]) -> Grid:
    return np.asarray(raw, dtype=np.int64)


def train_pairs(task: dict[str, Any]) -> list[tuple[Grid, Grid]]:
    return [(grid(ex["input"]), grid(ex["output"])) for ex in task.get("train", [])]


def validation_pairs(task: dict[str, Any], arc_gen_sample: int = 20) -> list[tuple[str, Grid, Grid]]:
    rows: list[tuple[str, Grid, Grid]] = []
    for split in ["train", "test"]:
        for idx, ex in enumerate(task.get(split, [])):
            rows.append((f"{split}{idx}", grid(ex["input"]), grid(ex["output"])))
    for idx, ex in enumerate(task.get("arc-gen", [])[:arc_gen_sample]):
        rows.append((f"arc{idx}", grid(ex["input"]), grid(ex["output"])))
    return rows


def colors(arr: Grid) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in np.unique(arr)))


def transform(arr: Grid, name: str) -> Grid:
    if name == "identity":
        return arr.copy()
    if name == "flip_h":
        return np.flip(arr, axis=1).copy()
    if name == "flip_v":
        return np.flip(arr, axis=0).copy()
    if name == "rot180":
        return np.flip(np.flip(arr, axis=0), axis=1).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "rot90_cw":
        return np.rot90(arr, k=3).copy()
    if name == "rot90_ccw":
        return np.rot90(arr, k=1).copy()
    if name == "anti_transpose":
        return np.flip(np.flip(arr.T, axis=0), axis=1).copy()
    raise ValueError(name)


def infer_color_map(pairs: list[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for x, y in pairs:
        if x.shape != y.shape:
            return None
        for src, dst in zip(x.ravel(), y.ravel()):
            src_i, dst_i = int(src), int(dst)
            if src_i in mapping and mapping[src_i] != dst_i:
                return None
            mapping[src_i] = dst_i
    return mapping


def apply_color_map(arr: Grid, mapping: dict[int, int]) -> Grid:
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == src] = dst
    return out


def validate_rule(rule: RuleFn, task: dict[str, Any], arc_gen_sample: int = 20) -> tuple[str, int, int, str]:
    passed = 0
    failed = 0
    for label, x, y in validation_pairs(task, arc_gen_sample):
        try:
            pred = rule(x)
        except Exception as exc:
            return "runtime_error", passed, failed + 1, f"{label}: {str(exc)[:120]}"
        if pred.shape != y.shape:
            return "fail", passed, failed + 1, f"{label}: shape {pred.shape}!={y.shape}"
        if not np.array_equal(pred, y):
            return "fail", passed, failed + 1, f"{label}: mismatch"
        passed += 1
    return "pass", passed, failed, "ok"


def common_shape_stats(pairs: list[tuple[Grid, Grid]]) -> dict[str, Any]:
    shape_pairs = Counter(f"{x.shape}->{y.shape}" for x, y in pairs)
    same_shape = all(x.shape == y.shape for x, y in pairs)
    changed_ratios = [float(np.sum(x != y) / max(1, x.size)) for x, y in pairs if x.shape == y.shape]
    return {
        "shape_pairs": ";".join(f"{k}:{v}" for k, v in sorted(shape_pairs.items())),
        "same_shape": same_shape,
        "avg_changed_ratio": float(sum(changed_ratios) / len(changed_ratios)) if changed_ratios else "",
        "max_changed_ratio": max(changed_ratios) if changed_ratios else "",
    }


def fit_color_map(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    mapping = infer_color_map(train_pairs(task))
    if mapping is None or all(src == dst for src, dst in mapping.items()):
        return "color_map", None, "no nontrivial consistent color map", 0
    return "color_map", lambda x: apply_color_map(x, mapping), f"mapping={mapping}", 1800 + 120 * len(mapping)


def fit_global_transform_color_map(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    pairs = train_pairs(task)
    transforms = ["identity", "flip_h", "flip_v", "rot180", "transpose", "rot90_cw", "rot90_ccw", "anti_transpose"]
    for name in transforms:
        transformed_pairs: list[tuple[Grid, Grid]] = []
        ok = True
        for x, y in pairs:
            tx = transform(x, name)
            if tx.shape != y.shape:
                ok = False
                break
            transformed_pairs.append((tx, y))
        if not ok:
            continue
        mapping = infer_color_map(transformed_pairs)
        if mapping is None:
            continue
        if name == "identity" and all(src == dst for src, dst in mapping.items()):
            continue
        return (
            "global_transform_color_map",
            lambda x, n=name, m=mapping: apply_color_map(transform(x, n), m),
            f"transform={name},mapping={mapping}",
            2200 + 200 * len(mapping),
        )
    return "global_transform_color_map", None, "no transform+color map fits train", 0


def fit_constant_output(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    outputs = [y for _, y in train_pairs(task)]
    if not outputs:
        return "constant_output", None, "no train examples", 0
    first = outputs[0]
    if any(not np.array_equal(first, y) for y in outputs[1:]):
        return "constant_output", None, "train outputs differ", 0
    active = int(np.count_nonzero(first))
    return "constant_output", lambda _x, y=first.copy(): y.copy(), f"shape={first.shape},nonzero={active}", 900 + 20 * max(1, active)


def crop_color_map_for(pairs: list[tuple[Grid, Grid]], top: int, left: int, height: int, width: int) -> dict[int, int] | None:
    transformed: list[tuple[Grid, Grid]] = []
    for x, y in pairs:
        if top + height > x.shape[0] or left + width > x.shape[1] or y.shape != (height, width):
            return None
        transformed.append((x[top : top + height, left : left + width], y))
    return infer_color_map(transformed)


def fit_fixed_crop(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    pairs = train_pairs(task)
    candidates: set[tuple[int, int, int, int]] | None = None
    for x, y in pairs:
        oh, ow = y.shape
        if oh > x.shape[0] or ow > x.shape[1]:
            return "fixed_crop_color_map", None, "output larger than input", 0
        local: set[tuple[int, int, int, int]] = set()
        for top in range(x.shape[0] - oh + 1):
            for left in range(x.shape[1] - ow + 1):
                mapping = crop_color_map_for([(x, y)], top, left, oh, ow)
                if mapping is not None:
                    local.add((top, left, oh, ow))
        if not local:
            return "fixed_crop_color_map", None, "no crop/color map fits train example", 0
        candidates = local if candidates is None else candidates.intersection(local)
        if not candidates:
            return "fixed_crop_color_map", None, "no common crop/color map", 0
    if not candidates:
        return "fixed_crop_color_map", None, "no candidates", 0
    for top, left, height, width in sorted(candidates):
        mapping = crop_color_map_for(pairs, top, left, height, width)
        if mapping is None:
            continue
        return (
            "fixed_crop_color_map",
            lambda x, t=top, l=left, h=height, w=width, m=mapping: apply_color_map(x[t : t + h, l : l + w], m),
            f"top={top},left={left},height={height},width={width},mapping={mapping}",
            1600 + 120 * len(mapping) + height * width,
        )
    return "fixed_crop_color_map", None, "no final crop mapping", 0


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def fit_bbox_crop(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    pairs = train_pairs(task)
    mask_specs: list[tuple[str, int | None]] = [("nonzero", None)]
    all_colors = sorted({int(v) for x, _ in pairs for v in np.unique(x) if int(v) != 0})
    mask_specs.extend(("color", color) for color in all_colors)
    for kind, color in mask_specs:
        transformed: list[tuple[Grid, Grid]] = []
        ok = True
        shapes: set[tuple[int, int]] = set()
        for x, y in pairs:
            mask = x != 0 if kind == "nonzero" else x == int(color)
            box = bbox(mask)
            if box is None:
                ok = False
                break
            r0, c0, r1, c1 = box
            crop = x[r0:r1, c0:c1]
            shapes.add(crop.shape)
            if crop.shape != y.shape:
                ok = False
                break
            transformed.append((crop, y))
        if not ok:
            continue
        mapping = infer_color_map(transformed)
        if mapping is None:
            continue

        def rule(x: Grid, k: str = kind, col: int | None = color, m: dict[int, int] = mapping) -> Grid:
            mask = x != 0 if k == "nonzero" else x == int(col)
            box = bbox(mask)
            if box is None:
                return x[:0, :0].copy()
            r0, c0, r1, c1 = box
            return apply_color_map(x[r0:r1, c0:c1], m)

        return (
            "bbox_crop_color_map",
            rule,
            f"mask={kind if color is None else f'color_{color}'},shapes={sorted(shapes)},mapping={mapping}",
            6500 + 200 * len(mapping),
        )
    return "bbox_crop_color_map", None, "no bbox crop fits train", 0


def fit_fixed_sparse_edit(task: dict[str, Any]) -> tuple[str, RuleFn | None, str, int]:
    pairs = train_pairs(task)
    expected: dict[tuple[int, int], int] = {}
    for x, y in pairs:
        if x.shape != y.shape:
            return "fixed_sparse_edit", None, "shape mismatch", 0
        for r, c in zip(*np.where(x != y)):
            key = (int(r), int(c))
            color = int(y[r, c])
            if key in expected and expected[key] != color:
                return "fixed_sparse_edit", None, "edit conflict", 0
            expected[key] = color
    if not expected:
        return "fixed_sparse_edit", None, "no edits", 0
    if len(expected) > 80:
        return "fixed_sparse_edit", None, f"too many fixed edits: {len(expected)}", 0

    def rule(x: Grid, edits: dict[tuple[int, int], int] = expected) -> Grid:
        out = x.copy()
        for (r, c), color in edits.items():
            if r < out.shape[0] and c < out.shape[1]:
                out[r, c] = color
        return out

    return "fixed_sparse_edit", rule, f"fixed_cells={len(expected)}", 1800 + 60 * len(expected)


def fit_rules(task: dict[str, Any]) -> list[tuple[str, RuleFn | None, str, int]]:
    return [
        fit_color_map(task),
        fit_global_transform_color_map(task),
        fit_constant_output(task),
        fit_fixed_crop(task),
        fit_bbox_crop(task),
        fit_fixed_sparse_edit(task),
    ]


def mine_task(queue_row: dict[str, str], task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_id = int(queue_row["task_id"])
    current_cost = int(float(queue_row["current_cost"]))
    pairs = train_pairs(task)
    stats = common_shape_stats(pairs)
    rows: list[dict[str, Any]] = []
    for family, rule, proof, expected_cost in fit_rules(task):
        if rule is None:
            rows.append(
                {
                    "task_id": task_id,
                    "rule_family": family,
                    "fit_status": "skipped",
                    "validation_status": "not_run",
                    "passed": 0,
                    "failed": 0,
                    "current_cost": current_cost,
                    "expected_cost_upper_bound": "",
                    "projected_gain": "",
                    "priority": "",
                    "proof": proof,
                    "lowering_plan": "",
                    "leakage_risk": "low: train-only fit",
                    "private_risk": "unknown",
                }
            )
            continue
        status, passed, failed, reason = validate_rule(rule, task, 20)
        projected_gain = point(expected_cost) - point(current_cost) if status == "pass" and expected_cost < current_cost else 0.0
        if family == "bbox_crop_color_map":
            lowering = "dynamic bbox via cheap reductions; requires exp029 low-cost lowering before ONNX"
            private = "medium-high"
        elif family == "fixed_crop_color_map":
            lowering = "constant Slice + optional 1x1 Conv"
            private = "medium"
        elif family == "global_transform_color_map":
            lowering = "Gather/Transpose + optional 1x1 Conv"
            private = "medium"
        elif family == "fixed_sparse_edit":
            lowering = "small coordinate ScatterND or constant mask Where after cost gate"
            private = "medium"
        elif family == "constant_output":
            lowering = "constant sparse output only if validation pass and tiny"
            private = "high"
        else:
            lowering = "1x1 Conv or small Equal/Where"
            private = "medium"
        rows.append(
            {
                "task_id": task_id,
                "rule_family": family,
                "fit_status": "fit",
                "validation_status": f"{status}:{passed}_pass_{failed}_fail",
                "passed": passed,
                "failed": failed,
                "current_cost": current_cost,
                "expected_cost_upper_bound": expected_cost,
                "projected_gain": f"{projected_gain:.6f}",
                "priority": f"{projected_gain * 10.0 + math.log(max(1, current_cost)):.6f}" if projected_gain > 0 else "0.000000",
                "proof": proof if status == "pass" else f"{proof}; validation={reason}",
                "lowering_plan": lowering,
                "leakage_risk": "low: train-only fit, validation uses test/arc-gen sample",
                "private_risk": private,
            }
        )
    summary = {
        "task_id": task_id,
        "current_cost": current_cost,
        "shape_pairs": stats["shape_pairs"],
        "same_shape": stats["same_shape"],
        "avg_changed_ratio": stats["avg_changed_ratio"],
        "max_changed_ratio": stats["max_changed_ratio"],
        "input_colors": " ".join(str(c) for c in sorted({v for x, _ in pairs for v in colors(x)})),
        "output_colors": " ".join(str(c) for c in sorted({v for _, y in pairs for v in colors(y)})),
        "pass_rule_count": sum(1 for row in rows if str(row["validation_status"]).startswith("pass:")),
        "best_projected_gain": max([float(row["projected_gain"] or 0.0) for row in rows] or [0.0]),
    }
    return rows, summary


def notes_text(result: dict[str, Any]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "signature lookup 196件の一部は、train-onlyで抽出できる固定crop、global transform、bbox crop、固定sparse editへ圧縮できる。",
        "この段階ではONNXを出さず、rule候補とproof logを作って次のlowering実験へ渡す。",
        "",
        "## Result",
        "",
        f"- scanned signature lookup tasks: `{result['scanned_task_count']}`",
        f"- validation-pass rule candidates: `{result['pass_rule_candidate_count']}`",
        f"- tasks with at least one pass rule: `{result['tasks_with_pass_rule_count']}`",
        f"- positive projected candidates: `{result['positive_projected_candidate_count']}`",
        f"- base local estimate: `{result['base_local_estimate']:.6f}`",
        "",
        "## Interpretation",
        "",
        "fitにはtrainだけを使い、testとarc-gen先頭20件はvalidation専用にした。"
        "これにより、exp016系のarc-gen memorized lookupから説明可能ruleへ戻す入口を作った。",
        "",
        "## Next",
        "",
        "- passした `fixed_crop_color_map` / `global_transform_color_map` は低cost ONNXへ即接続する。",
        "- `bbox_crop_color_map` は有望だがdynamic bbox loweringが課題なので、exp029でcost-aware loweringを作る。",
        "- projected gainが正の候補だけを、次のONNX campaignへ渡す。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_result = load_json(BASE_RESULT_PATH)
    base_local = float(base_result.get("new_local_estimate") or base_result.get("local_estimate"))
    queue = load_csv(QUEUE_PATH)
    lookup_rows = [row for row in queue if row["synthesis_family"] == "signature_lookup_current"]
    candidate_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for row in lookup_rows:
        task_id = int(row["task_id"])
        task = load_json(DATA_DIR / f"task{task_id:03d}.json")
        candidates, summary = mine_task(row, task)
        candidate_rows.extend(candidates)
        task_rows.append(summary)

    candidate_rows.sort(key=lambda row: (-float(row["projected_gain"] or 0.0), int(row["task_id"]), str(row["rule_family"])))
    task_rows.sort(key=lambda row: (-float(row["best_projected_gain"]), -int(row["current_cost"]), int(row["task_id"])))
    write_csv(EXP_DIR / "rule_candidate_manifest.csv", candidate_rows)
    write_csv(EXP_DIR / "task_rule_summary.csv", task_rows)

    pass_rows = [row for row in candidate_rows if str(row["validation_status"]).startswith("pass:")]
    positive_rows = [row for row in pass_rows if float(row["projected_gain"] or 0.0) > 0.0]
    family_counts = Counter(row["rule_family"] for row in pass_rows)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_mining_ready",
        "base_exp": "experiments/exp023_graph_surgery_exp016",
        "base_local_estimate": base_local,
        "gap_to_6500": 6500.0 - base_local,
        "gap_to_7600": 7600.0 - base_local,
        "scanned_task_count": len(lookup_rows),
        "candidate_rows": len(candidate_rows),
        "pass_rule_candidate_count": len(pass_rows),
        "tasks_with_pass_rule_count": len({int(row["task_id"]) for row in pass_rows}),
        "positive_projected_candidate_count": len(positive_rows),
        "pass_family_counts": dict(sorted(family_counts.items())),
        "top_positive_candidates": positive_rows[:30],
        "outputs": {
            "rule_candidate_manifest": "rule_candidate_manifest.csv",
            "task_rule_summary": "task_rule_summary.csv",
        },
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: miner only, no selected ONNX bundle generated.",
        "leakage_risk": "medium-low: rules are fit on train only, but candidate ranking uses local validation.",
        "overfitting_risk": "medium: arc-gen sample validation is not a private guarantee.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

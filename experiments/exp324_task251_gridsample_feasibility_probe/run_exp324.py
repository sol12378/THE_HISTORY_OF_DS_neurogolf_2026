from __future__ import annotations

import json
import pathlib
import sys
import time
import zipfile
from collections import deque

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from phase1_rewrite_utils import (  # noqa: E402
    examples_for,
    infer_static_ok,
    load_neurogolf_utils,
    load_task,
    point,
    score_model,
    sha256,
    validate_examples,
)


EXP_ID = "exp324_task251_gridsample_feasibility_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 251
BASE_ZIP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates" / "submission.zip"
BASE_COST = 100580
BASE_LB = 6008.96


def grid_to_array(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def closed_zero_component_neighbor2_to_1(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    seen = np.zeros((h, w), dtype=bool)
    for sr in range(h):
        for sc in range(w):
            if out[sr, sc] != 0 or seen[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            boundary_colors: set[int] = set()
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if nr < 0 or nc < 0 or nr >= h or nc >= w:
                        continue
                    color = int(out[nr, nc])
                    if color == 0 and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif color != 0:
                        boundary_colors.add(color)
            if not touches_border and boundary_colors == {2}:
                for r, c in cells:
                    out[r, c] = 1
    return out


def validate_python_rule() -> dict[str, object]:
    task = load_task(TASK_ID)
    rows: dict[str, dict[str, int]] = {}
    total_pass = 0
    total_fail = 0
    for split_name, examples in (
        ("train", task["train"]),
        ("test", task["test"]),
        ("arc-gen", task["arc-gen"]),
    ):
        passed = 0
        failed = 0
        for ex in examples:
            pred = closed_zero_component_neighbor2_to_1(grid_to_array(ex["input"]))
            expected = grid_to_array(ex["output"])
            if np.array_equal(pred, expected):
                passed += 1
            else:
                failed += 1
        rows[split_name] = {"pass": passed, "fail": failed}
        total_pass += passed
        total_fail += failed
    return {
        "rule": "closed_zero_component_neighbor2_to_1",
        "splits": rows,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "status": f"{total_pass}_pass_{total_fail}_fail",
    }


def make_identity_gridsample_raw(opset: int) -> bytes:
    xs = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    grid = np.zeros((1, 30, 30, 2), dtype=np.float32)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            grid[0, r, c, 0] = x
            grid[0, r, c, 1] = y
    node = helper.make_node(
        "GridSample",
        ["input", "identity_grid"],
        ["output"],
        mode="nearest",
        padding_mode="zeros",
        align_corners=1,
    )
    graph = helper.make_graph(
        [node],
        f"{EXP_ID}_opset{opset}_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(grid, "identity_grid")],
    )
    model = helper.make_model(
        graph,
        producer_name=f"{EXP_ID}_opset{opset}",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", opset)],
    )
    return model.SerializeToString()


def evaluate_gridsample_variant(utils, opset: int) -> dict[str, object]:
    raw = make_identity_gridsample_raw(opset)
    row: dict[str, object] = {
        "name": f"identity_gridsample_opset{opset}",
        "opset": opset,
        "file_bytes": len(raw),
        "sha256": sha256(raw),
    }
    try:
        model = onnx.load_model_from_string(raw)
        static_ok, static_reason = infer_static_ok(model)
        row["static_ok"] = static_ok
        row["static_reason"] = static_reason
    except Exception as exc:
        row["static_ok"] = False
        row["static_reason"] = f"parse failed: {str(exc)[:180]}"
        return row
    if not row["static_ok"]:
        return row

    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        row["sanitize_ok"] = False
        row["sanitize_reason"] = "sanitize_model returned None"
        return row
    row["sanitize_ok"] = True
    sanitized_raw = sanitized.SerializeToString()
    row["sanitized_file_bytes"] = len(sanitized_raw)
    row["sanitized_sha256"] = sha256(sanitized_raw)

    ok, reason, passed, failed = validate_examples(utils, sanitized_raw, TASK_ID, -1)
    row["validation_ok"] = ok
    row["validation_status"] = f"{passed}_pass_{failed}_fail"
    row["validation_reason"] = reason

    memory, params, score_reason = score_model(utils, sanitized_raw, TASK_ID, f"identity_gridsample_opset{opset}", EXP_DIR)
    row["score_reason"] = score_reason
    if memory is not None and params is not None:
        cost = memory + params
        row["memory"] = memory
        row["params"] = params
        row["cost"] = cost
        row["points"] = point(cost)
        row["delta_if_correct_vs_base"] = point(cost) - point(BASE_COST)
        row["expected_lb_if_correct"] = BASE_LB + row["delta_if_correct_vs_base"]
    return row


def main() -> None:
    start = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    task = load_task(TASK_ID)

    with zipfile.ZipFile(BASE_ZIP) as zf:
        base_raw = zf.read(f"task{TASK_ID:03d}.onnx")
    base_memory, base_params, base_reason = score_model(utils, base_raw, TASK_ID, "exp297_base", EXP_DIR)

    shapes = sorted({(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) for ex in examples_for(task, -1)})
    changed_counts = [
        int(np.sum(grid_to_array(ex["input"]) != grid_to_array(ex["output"])))
        for ex in examples_for(task, -1)
    ]
    input_has_color1 = [
        bool(np.any(grid_to_array(ex["input"]) == 1))
        for ex in examples_for(task, -1)
    ]

    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "feasibility_probe_complete",
        "task_id": TASK_ID,
        "hypothesis": "task251 GridSample lane should first be real-scored; pure data movement may be cheap but cannot by itself create the color1 fill mask.",
        "base": {
            "source": str(BASE_ZIP.relative_to(ROOT)),
            "cost_from_summary": BASE_COST,
            "score_reason": base_reason,
            "memory": base_memory,
            "params": base_params,
            "cost_rescored": (base_memory + base_params) if base_memory is not None and base_params is not None else None,
            "points_rescored": point(base_memory + base_params) if base_memory is not None and base_params is not None else None,
        },
        "data_profile": {
            "example_count": len(examples_for(task, -1)),
            "shapes": shapes,
            "changed_cells_min": min(changed_counts),
            "changed_cells_mean": float(np.mean(changed_counts)),
            "changed_cells_max": max(changed_counts),
            "input_has_color1_count": int(sum(input_has_color1)),
            "input_lacks_color1_count": int(len(input_has_color1) - sum(input_has_color1)),
        },
        "python_rule_validation": validate_python_rule(),
        "gridsample_variants": [
            evaluate_gridsample_variant(utils, 16),
            evaluate_gridsample_variant(utils, 20),
        ],
        "decision": "",
        "submission_decision": "no_submit: feasibility probe only; identity GridSample is intentionally not task-correct.",
        "leakage_risk": "low: uses only official task examples for rule revalidation and an identity primitive probe; no hidden/public feedback.",
        "overfitting_risk": "low-to-medium: task251 rule is task-specific but already full-arc validated; this experiment does not submit a replacement.",
        "runtime_seconds": None,
    }

    successful_costs = [
        row["cost"]
        for row in result["gridsample_variants"]
        if isinstance(row, dict) and isinstance(row.get("cost"), int)
    ]
    if successful_costs:
        best_cost = min(successful_costs)
        if result["data_profile"]["input_lacks_color1_count"] > 0:
            result["decision"] = (
                f"GridSample is official-scoreable with best observed identity cost {best_cost}, "
                "but pure sampling is insufficient for task251 because many inputs lack color1 while outputs create color1 fills. "
                "Next candidate must generate a closed-component mask and combine it with a cheap color1/recolor primitive, or pivot to another GridSample target."
            )
        else:
            result["decision"] = (
                f"GridSample is official-scoreable with best observed identity cost {best_cost}; "
                "continue to a mask-generation candidate."
            )
    else:
        result["decision"] = "GridSample could not be statically scored in this environment; do not spend more task251 work before resolving primitive support."

    result["runtime_seconds"] = time.time() - start
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## 目的

task251 GridSample queue の最初の score-direct probe として、実ONNX `GridSample` が NeuroGolf official scorer で使えるか、また純粋な data movement だけで task251 を解ける余地があるか確認する。

## 結果

- Python rule: `{result['python_rule_validation']['status']}`
- base cost(rescored): `{result['base']['cost_rescored']}`
- GridSample variants:
  - opset16: `{result['gridsample_variants'][0]}`
  - opset20: `{result['gridsample_variants'][1]}`
- input lacks color1 examples: `{result['data_profile']['input_lacks_color1_count']}` / `{result['data_profile']['example_count']}`

## 判断

{result['decision']}

## Submission

`{result['submission_decision']}`

## Risk

- leakage risk: {result['leakage_risk']}
- overfitting risk: {result['overfitting_risk']}
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()

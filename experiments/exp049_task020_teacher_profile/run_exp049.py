from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter
from datetime import date
from typing import Any

import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_neurogolf_utils, load_task, sha256


EXP_ID = "exp049_task020_teacher_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20
STRICT_EXP = ROOT / "experiments" / "exp005_top_cost_rewrite_strict"
TEACHER_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"


def read_zip_raw(exp_dir: pathlib.Path, task_id: int) -> bytes:
    with zipfile.ZipFile(exp_dir / "submission.zip") as zf:
        return zf.read(f"task{task_id:03d}.onnx")


def profile_raw(utils: Any, label: str, raw: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(raw)
    sanitized = utils.sanitize_model(model)
    if sanitized is None:
        raise RuntimeError(f"sanitize failed: {label}")
    sanitized_raw = sanitized.SerializeToString()

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = str(EXP_DIR / f"profile_{label}_task{TARGET_TASK:03d}")
    session = ort.InferenceSession(sanitized_raw, options, providers=["CPUExecutionProvider"])
    benchmark = utils.convert_to_numpy(examples_for(load_task(TARGET_TASK), 1)[0])
    utils.run_network(session, benchmark["input"])
    trace_path = pathlib.Path(session.end_profiling())
    memory, params = utils.score_network(sanitized, str(trace_path))

    node_rows = []
    op_counts = Counter()
    for idx, node in enumerate(sanitized.graph.node):
        op_counts[node.op_type] += 1
        node_rows.append(
            {
                "label": label,
                "idx": idx,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": "|".join(node.input),
                "outputs": "|".join(node.output),
                "attr_count": len(node.attribute),
            }
        )

    init_rows = []
    for init in sanitized.graph.initializer:
        elems = 1
        for dim in init.dims:
            elems *= int(dim)
        init_rows.append({"label": label, "name": init.name, "dims": "x".join(str(dim) for dim in init.dims), "elem_count": elems, "data_type": init.data_type})
    init_rows = sorted(init_rows, key=lambda row: (-int(row["elem_count"]), row["name"]))

    trace_rows = []
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        for event in trace:
            if event.get("cat") != "Node":
                continue
            args = event.get("args", {})
            trace_rows.append(
                {
                    "label": label,
                    "name": str(event.get("name", "")).replace("_kernel_time", ""),
                    "dur": event.get("dur", ""),
                    "output_type_shape": json.dumps(args.get("output_type_shape", []), ensure_ascii=False),
                }
            )
        trace_path.unlink(missing_ok=True)

    return {
        "label": label,
        "raw": sanitized_raw,
        "memory": memory,
        "params": params,
        "cost": None if memory is None or params is None else memory + params,
        "sha256": sha256(sanitized_raw),
        "node_count": len(sanitized.graph.node),
        "initializer_count": len(sanitized.graph.initializer),
        "initializer_elem_total": sum(int(row["elem_count"]) for row in init_rows),
        "op_counts": dict(op_counts),
        "largest_initializers": init_rows[:10],
        "node_rows": node_rows,
        "init_rows": init_rows,
        "trace_rows": trace_rows,
    }


def write_rows(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    strict = profile_raw(utils, "strict_exp005", read_zip_raw(STRICT_EXP, TARGET_TASK))
    teacher = profile_raw(utils, "teacher_exp023", read_zip_raw(TEACHER_EXP, TARGET_TASK))

    write_rows(EXP_DIR / "node_inventory.csv", strict["node_rows"] + teacher["node_rows"])
    write_rows(EXP_DIR / "initializer_inventory.csv", strict["init_rows"] + teacher["init_rows"])
    write_rows(EXP_DIR / "trace_node_shapes.csv", strict["trace_rows"] + teacher["trace_rows"])

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "diagnostic_complete",
        "target_task": TARGET_TASK,
        "strict": {k: v for k, v in strict.items() if k not in {"raw", "node_rows", "init_rows", "trace_rows"}},
        "teacher": {k: v for k, v in teacher.items() if k not in {"raw", "node_rows", "init_rows", "trace_rows"}},
        "cost_delta_teacher_vs_strict": teacher["cost"] - strict["cost"],
        "point_gain_teacher_vs_strict": max(1.0, 25.0 - __import__("math").log(teacher["cost"])) - max(1.0, 25.0 - __import__("math").log(strict["cost"])),
        "interpretation": "Teacher is a compact signature lookup artifact; use it as output oracle/structure hint only, not as submit-safe solution.",
        "next": "Mine task020 input-output rule from examples and teacher outputs; reject per-signature memorization.",
        "runtime_seconds": time.time() - started,
        "leakage_risk": "diagnostic: teacher is high-risk signature lookup.",
        "overfitting_risk": "diagnostic: no candidate emitted.",
    }
    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    notes = f"""# {EXP_ID}

## 目的

exp048のP0最大task020について、submit-safe strict artifactとhigh-risk teacher artifactを比較profileする。

## 結果

- strict cost: {strict["cost"]}
- teacher cost: {teacher["cost"]}
- teacher point gain: {result["point_gain_teacher_vs_strict"]:.6f}
- strict op counts: {strict["op_counts"]}
- teacher op counts: {teacher["op_counts"]}
- teacher largest initializers: {teacher["largest_initializers"][:5]}

## 解釈

teacherは安いが `signature_scatternd_lookup` 由来なので、そのまま提出戦略には使わない。task020の明示ruleを掘るためのteacher/oracleとして使う。

## 次

task020の入出力例を可視/特徴化し、changed-cell maskやobject completion ruleを探索する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import importlib.util
import json
import pathlib
import sys
import time
import zipfile
from typing import Any

import numpy as np
import onnxruntime as ort


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp004_public_blend_relaxed_static"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
SUBMISSION_ZIP = EXP_DIR / "submission.zip"
DETAIL_PATH = EXP_DIR / "full_validation_manifest.csv"
RESULT_PATH = EXP_DIR / "result.json"


def load_neurogolf_utils() -> Any:
    path = DATA_DIR / "neurogolf_utils" / "neurogolf_utils.py"
    spec = importlib.util.spec_from_file_location("neurogolf_utils_exp004_full", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["neurogolf_utils_exp004_full"] = module
    spec.loader.exec_module(module)
    return module


def load_task_examples(task_id: int) -> list[dict[str, Any]]:
    task_path = DATA_DIR / f"task{task_id:03d}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    return task["train"] + task["test"] + task["arc-gen"]


def validate_task(utils: Any, task_id: int, raw: bytes) -> dict[str, Any]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
    examples = load_task_examples(task_id)
    passed = 0
    failed = 0
    first_error = ""
    start = time.time()
    for idx, example in enumerate(examples):
        benchmark = utils.convert_to_numpy(example)
        if benchmark is None:
            continue
        try:
            output = utils.run_network(session, benchmark["input"])
        except Exception as exc:
            failed += 1
            if not first_error:
                first_error = f"runtime example {idx}: {str(exc)[:160]}"
            continue
        if np.array_equal(output, benchmark["output"]):
            passed += 1
        else:
            failed += 1
            if not first_error:
                first_error = f"mismatch example {idx}"
    return {
        "task_id": task_id,
        "status": "pass" if failed == 0 else "fail",
        "examples": passed + failed,
        "passed": passed,
        "failed": failed,
        "first_error": first_error,
        "seconds": round(time.time() - start, 4),
    }


def main() -> None:
    utils = load_neurogolf_utils()
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(SUBMISSION_ZIP) as zf:
        names = sorted(zf.namelist())
        for idx, name in enumerate(names, start=1):
            if idx % 25 == 0:
                print(f"Validating selected {idx}/{len(names)}", flush=True)
            task_id = int(pathlib.Path(name).stem.replace("task", ""))
            rows.append(validate_task(utils, task_id, zf.read(name)))

    with DETAIL_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "status", "examples", "passed", "failed", "first_error", "seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    failed_tasks = [row["task_id"] for row in rows if row["status"] != "pass"]
    result["full_arc_gen_validation"] = {
        "status": "pass" if not failed_tasks else "fail",
        "task_count": len(rows),
        "pass_task_count": len(rows) - len(failed_tasks),
        "fail_task_count": len(failed_tasks),
        "failed_tasks": failed_tasks,
        "total_examples": sum(int(row["examples"]) for row in rows),
        "total_failed_examples": sum(int(row["failed"]) for row in rows),
        "detail": str(DETAIL_PATH.relative_to(ROOT)),
    }
    if failed_tasks:
        result["status"] = "full_validation_failed"
    elif float(result.get("local_estimate", 0)) >= 6500:
        result["status"] = "submit_candidate_full_validated"
    else:
        result["status"] = "full_validation_pass_below_target"
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["full_arc_gen_validation"], ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

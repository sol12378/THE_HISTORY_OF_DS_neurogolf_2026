from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import date

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, sha256, validate_examples  # noqa: E402


EXP_ID = "exp_b029_task251_reachability_depth_surgery"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
TASK_ID = 251


@dataclass(frozen=True)
class SurgeryRow:
    replacement: str
    removed_depth_hint: int
    validation_status: str
    status: str
    reason: str
    sha256: str


def replace_input(model: onnx.ModelProto, old: str, new: str) -> None:
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        raw = zf.read(f"task{TASK_ID:03d}.onnx")
    base_model = onnx.load_from_string(raw)

    replacements = [
        ("safe_name_58", 1),
        ("safe_name_55", 2),
        ("safe_name_52", 3),
        ("safe_name_49", 4),
        ("safe_name_46", 5),
        ("safe_name_43", 6),
        ("safe_name_40", 7),
    ]
    rows: list[SurgeryRow] = []
    for repl, removed_depth in replacements:
        model = onnx.ModelProto()
        model.CopyFrom(base_model)
        replace_input(model, "safe_name_61", repl)
        data = model.SerializeToString()
        ok, reason, passed, failed = validate_examples(utils, data, TASK_ID, -1)
        rows.append(
            SurgeryRow(
                replacement=repl,
                removed_depth_hint=removed_depth,
                validation_status=f"{passed}_pass_{failed}_fail",
                status="valid" if ok else "rejected",
                reason=reason,
                sha256=sha256(data),
            )
        )

    with (EXP_DIR / "surgery_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SurgeryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    valid = [row for row in rows if row.status == "valid"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "no_valid_depth_reduction" if not valid else "valid_depth_reduction_found",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_id": TASK_ID,
        "candidate_count": len(rows),
        "valid_count": len(valid),
        "rows": [asdict(row) for row in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit",
        "runtime_seconds": time.time() - started,
        "leakage_risk": "low: graph surgery validation only; no output labels used to construct a lookup.",
        "overfitting_risk": "low-to-medium: all candidates rejected before any submission.",
        "decision": "Existing task251 reachability depth appears necessary on arc-gen; pursue alternate closed-form/rectangle lowering rather than depth pruning.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task251既存artifactのreachability depthを削れるか確認する。

## 結果

- candidates: `{len(rows)}`
- valid depth reductions: `{len(valid)}`
- best partial: `{rows[0].replacement}` -> `{rows[0].validation_status}`

## 判断

depth削減は全候補reject。既存artifactのreachability depthはarc-gen上ほぼ必要。次は深さを削るのではなく、closed-form/rectangle-specific maskへ表現を変える。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

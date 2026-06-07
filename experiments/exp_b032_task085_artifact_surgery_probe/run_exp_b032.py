from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date

import onnx

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, sha256, validate_examples  # noqa: E402


EXP_ID = "exp_b032_task085_artifact_surgery_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
TASK_ID = 85


@dataclass(frozen=True)
class SurgeryRow:
    candidate: str
    validation_status: str
    status: str
    reason: str
    sha256: str


def bypass_cast(model: onnx.ModelProto, remove_node: bool) -> onnx.ModelProto:
    out = onnx.ModelProto()
    out.CopyFrom(model)
    for node in out.graph.node:
        for i, inp in enumerate(node.input):
            if inp == "safe_name_14":
                node.input[i] = "input"
    if remove_node:
        keep = [node for node in out.graph.node if not (node.op_type == "Cast" and node.output and node.output[0] == "safe_name_14")]
        del out.graph.node[:]
        out.graph.node.extend(keep)
    return out


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        raw = zf.read(f"task{TASK_ID:03d}.onnx")
    base = onnx.load_from_string(raw)
    rows: list[SurgeryRow] = []
    for name, remove in [("cast_bypass_keep_node", False), ("cast_bypass_remove_node", True)]:
        model = bypass_cast(base, remove)
        data = model.SerializeToString()
        ok, reason, passed, failed = validate_examples(utils, data, TASK_ID, -1)
        rows.append(
            SurgeryRow(
                candidate=name,
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
        "status": "no_valid_surgery" if not valid else "valid_surgery_found",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "task_id": TASK_ID,
        "candidate_count": len(rows),
        "valid_count": len(valid),
        "rows": [asdict(row) for row in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit",
        "runtime_seconds": time.time() - started,
        "leakage_risk": "low: graph surgery validation only.",
        "overfitting_risk": "low: candidates rejected before scoring/submission.",
        "decision": "Task085 Cast is required for float16 internal graph; existing artifact is already compact. Continue only with stronger surgery or alternate lane.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

task085既存artifactから明らかなCastを削れるか確認する。

## 結果

- candidates: `{len(rows)}`
- valid: `{len(valid)}`
- reason: Cast bypassすると `safe_name_15` がfloat16期待なのにinput floatが流れ、ORT type error。

## 判断

task085 artifactはfloat16内部計算に依存しており、単純Cast削除は不可。rule hitはあるが、このartifactはかなりcompactなので、短期score目的なら別laneへ移る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

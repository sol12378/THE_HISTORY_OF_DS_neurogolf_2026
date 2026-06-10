from __future__ import annotations

import csv
import json
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp137_private_failure_consensus_audit"
EXP136_DIR = ROOT / "experiments" / "exp136_private_failure_subset_inventory"
BASE_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def task_tokens(task_string: str) -> list[int]:
    return [int(tok) for tok in task_string.split() if tok.strip()]


def model_profile(raw: bytes) -> dict[str, Any]:
    model = onnx.load_model_from_string(raw)
    ops = Counter(node.op_type for node in model.graph.node)
    initializer_bytes = 0
    for init in model.graph.initializer:
        elem_size = {
            onnx.TensorProto.FLOAT: 4,
            onnx.TensorProto.FLOAT16: 2,
            onnx.TensorProto.DOUBLE: 8,
            onnx.TensorProto.INT64: 8,
            onnx.TensorProto.INT32: 4,
            onnx.TensorProto.INT8: 1,
            onnx.TensorProto.UINT8: 1,
            onnx.TensorProto.BOOL: 1,
        }.get(init.data_type, 4)
        n = 1
        for dim in init.dims:
            n *= int(dim)
        initializer_bytes += n * elem_size
    return {
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "initializer_bytes": initializer_bytes,
        "top_ops": " ".join(f"{op}:{count}" for op, count in ops.most_common(8)),
        "has_matmul": int("MatMul" in ops),
        "has_scatternd": int("ScatterND" in ops),
        "has_gathernd": int("GatherND" in ops),
        "has_where": int("Where" in ops),
        "has_conv": int("Conv" in ops),
    }


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows = {int(row["task_id"]): row for row in read_csv(EXP136_DIR / "task_risk_inventory.csv")}
    subset_rows = read_csv(EXP136_DIR / "subset_candidates.csv")
    top_rows = subset_rows[:20]

    freq = Counter()
    exact_freq = Counter()
    weighted = Counter()
    for rank, row in enumerate(top_rows, start=1):
        tasks = task_tokens(row["tasks"])
        weight = 1.0 / rank
        exact = abs(float(row["gap_error"])) < 0.005
        for tid in tasks:
            freq[tid] += 1
            weighted[tid] += weight
            if exact:
                exact_freq[tid] += 1

    audit_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for tid, count in freq.most_common():
            inv = inventory_rows[tid]
            raw = zf.read(f"task{tid:03d}.onnx")
            prof = model_profile(raw)
            row: dict[str, Any] = {
                "task_id": tid,
                "top20_frequency": count,
                "exact_gap_frequency": exact_freq[tid],
                "weighted_frequency": round(weighted[tid], 6),
                "source": inv["source"],
                "route": inv["route"],
                "cost": inv["cost"],
                "points": inv["points"],
                "risk_score": inv["risk_score"],
                "input_shape_count": inv["input_shape_count"],
                "output_shape_count": inv["output_shape_count"],
                "max_input_area": inv["max_input_area"],
                "max_output_area": inv["max_output_area"],
                **prof,
            }
            audit_rows.append(row)

    audit_rows.sort(
        key=lambda r: (
            -int(r["exact_gap_frequency"]),
            -int(r["top20_frequency"]),
            -float(r["weighted_frequency"]),
            -float(r["risk_score"]),
            int(r["task_id"]),
        )
    )
    with (EXP_DIR / "consensus_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    queue = audit_rows[:15]
    result = {
        "exp_id": "exp137_private_failure_consensus_audit",
        "date": "2026-06-10",
        "status": "consensus_audit_ready",
        "source_exp": "exp136_private_failure_subset_inventory",
        "subset_rows_used": len(top_rows),
        "unique_candidate_tasks": len(audit_rows),
        "priority_queue": [
            {
                "task_id": row["task_id"],
                "top20_frequency": row["top20_frequency"],
                "exact_gap_frequency": row["exact_gap_frequency"],
                "source": row["source"],
                "route": row["route"],
                "points": float(row["points"]),
                "risk_score": float(row["risk_score"]),
                "top_ops": row["top_ops"],
            }
            for row in queue
        ],
        "outputs": {"consensus_audit": str((EXP_DIR / "consensus_audit.csv").relative_to(ROOT))},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Audit the top consensus tasks first; prioritize alternatives or robust repairs for high-frequency sparse/object and crop tasks before spending Kaggle bisection probes.",
        "leakage_risk": "low: model provenance and graph structure audit only.",
        "overfitting_risk": "medium: consensus comes from subset-sum candidates and remains indirect.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


def write_notes(result: dict[str, Any]) -> None:
    lines = [
        "# exp137_private_failure_consensus_audit",
        "",
        "## Hypothesis",
        "",
        "exp136の上位subset候補に頻出するtaskほど、-352 gapを作るprivate failure群である可能性が高い。",
        "",
        "## Result",
        "",
        f"- subset rows used: `{result['subset_rows_used']}`",
        f"- unique candidate tasks: `{result['unique_candidate_tasks']}`",
        "",
        "## Priority Queue",
        "",
    ]
    for row in result["priority_queue"]:
        lines.append(
            f"- task{int(row['task_id']):03d}: freq `{row['top20_frequency']}`, exact `{row['exact_gap_frequency']}`, "
            f"source `{row['source']}`, route `{row['route']}`, points `{float(row['points']):.4f}`, ops `{row['top_ops']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "上位候補は低番号taskに偏るため、subset-sumだけを信じず、頻度・source・graph構造を合わせて監査queueとして使う。最初のrepair候補は頻出かつ既にrule資産があるtask366、またはsource差し替え可能性が高いcrop/sparse taskを優先する。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "診断のみで提出なし。private failureの直接証明ではない。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

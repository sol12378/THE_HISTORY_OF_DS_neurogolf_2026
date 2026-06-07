from __future__ import annotations

import csv
import json
import pathlib
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

import onnx
import onnxruntime as ort

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.phase1_rewrite_utils import (
    ROOT,
    Candidate,
    evaluate_candidate,
    infer_static_ok,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    sha256,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_ID = "exp023_graph_surgery_exp016"
EXP_DIR = ROOT / "experiments" / EXP_ID
INPUT_EXP = ROOT / "experiments" / "exp016_top100_rewrite_campaign"
OUTPUT_ZIP = EXP_DIR / "submission.zip"


@dataclass
class SurgeryRow:
    task_id: int
    variant: str
    baseline_cost: int
    candidate_cost: int | str
    baseline_points: float
    candidate_points: float | str
    file_bytes: int | str
    validation_status: str
    status: str
    reason: str
    sha256: str


def remove_unused_initializers(model: onnx.ModelProto) -> int:
    used = {name for node in model.graph.node for name in node.input if name}
    kept = [init for init in model.graph.initializer if init.name in used]
    removed = len(model.graph.initializer) - len(kept)
    if removed:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
    return removed


def pruned_raw(raw: bytes) -> tuple[bytes | None, str]:
    try:
        model = onnx.load_model_from_string(raw)
    except Exception as exc:
        return None, f"parse failed: {str(exc)[:160]}"
    removed = remove_unused_initializers(model)
    if removed == 0:
        return None, "no unused initializers"
    ok, reason = infer_static_ok(model)
    if not ok:
        return None, reason
    return model.SerializeToString(), f"removed_initializers={removed}"


def ort_optimized_raw(raw: bytes, level: ort.GraphOptimizationLevel, task_id: int, label: str) -> tuple[bytes | None, str]:
    with tempfile.TemporaryDirectory(dir=EXP_DIR) as td:
        out = pathlib.Path(td) / f"task{task_id:03d}_{label}.onnx"
        options = ort.SessionOptions()
        options.optimized_model_filepath = str(out)
        options.graph_optimization_level = level
        try:
            session = ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])
            del session
        except Exception as exc:
            return None, f"opt session failed: {str(exc)[:160]}"
        if not out.exists():
            return None, "optimized model not written"
        return out.read_bytes(), "ok"


def candidate_variants(task_id: int, raw: bytes, route: str) -> list[Candidate]:
    variants: list[Candidate] = []
    pruned, reason = pruned_raw(raw)
    variants.append(Candidate(task_id, "surgery_prune_unused_initializers", route, pruned, "generated" if pruned else "skipped", reason))
    return variants


def selected_row(task_id: int, base: Any, record: SurgeryRow | None, raw: bytes) -> dict[str, Any]:
    if record is None:
        return {
            "task_id": task_id,
            "source": base.source,
            "template_name": base.template_name,
            "route": base.route,
            "cost": base.cost,
            "local_points": base.points,
            "file_bytes": len(base.raw),
            "status": "baseline",
            "reason": "no graph surgery gain",
            "sha256": sha256(base.raw),
        }
    return {
        "task_id": task_id,
        "source": f"{EXP_ID}_{record.variant}",
        "template_name": record.variant,
        "route": base.route,
        "cost": record.candidate_cost,
        "local_points": record.candidate_points,
        "file_bytes": len(raw),
        "status": "improved",
        "reason": record.reason,
        "sha256": sha256(raw),
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base_tasks = load_base_tasks(INPUT_EXP)
    final_raw = {task_id: base.raw for task_id, base in base_tasks.items()}
    candidate_rows: list[SurgeryRow] = []
    selected_rows: list[dict[str, Any]] = []
    improved: dict[int, SurgeryRow] = {}
    started = time.time()

    for idx, task_id in enumerate(sorted(base_tasks), start=1):
        if idx % 25 == 0:
            print(f"{EXP_ID} {idx}/400", flush=True)
        base = base_tasks[task_id]
        best_record: SurgeryRow | None = None
        best_raw: bytes | None = None
        try:
            variants = candidate_variants(task_id, base.raw, base.route)
        except Exception as exc:
            variants = [
                Candidate(
                    task_id,
                    "surgery_variant_generation",
                    base.route,
                    None,
                    "rejected",
                    f"variant generation failed: {str(exc)[:160]}",
                )
            ]
        for candidate in variants:
            eval_row, raw = evaluate_candidate(utils, candidate, base, 20, EXP_DIR)
            row = SurgeryRow(
                task_id=eval_row.task_id,
                variant=eval_row.template_name,
                baseline_cost=eval_row.baseline_cost,
                candidate_cost=eval_row.candidate_cost,
                baseline_points=eval_row.baseline_points,
                candidate_points=eval_row.candidate_points,
                file_bytes=eval_row.file_bytes,
                validation_status=eval_row.validation_status,
                status=eval_row.status,
                reason=eval_row.reason,
                sha256=eval_row.sha256,
            )
            candidate_rows.append(row)
            if raw is None or row.status != "improved" or not isinstance(row.candidate_cost, int):
                continue
            if best_record is None or int(row.candidate_cost) < int(best_record.candidate_cost):
                best_record = row
                best_raw = raw
        if best_record is not None and best_raw is not None:
            improved[task_id] = best_record
            final_raw[task_id] = best_raw

    write_zip(OUTPUT_ZIP, final_raw)
    for task_id in sorted(base_tasks):
        selected_rows.append(selected_row(task_id, base_tasks[task_id], improved.get(task_id), final_raw[task_id]))

    with (EXP_DIR / "candidate_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SurgeryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in candidate_rows])

    selected_fields = ["task_id", "source", "template_name", "route", "cost", "local_points", "file_bytes", "status", "reason", "sha256"]
    with (EXP_DIR / "selected_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    baseline_score = sum(task.points for task in base_tasks.values())
    new_score = sum(float(row["local_points"]) for row in selected_rows)
    status_counts = Counter(row.status for row in candidate_rows)
    variant_counts = Counter(row.variant for row in candidate_rows if row.status == "improved")
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "target_reached" if new_score >= 6500 else ("improved_below_target" if improved else "no_gain"),
        "base_exp": str(INPUT_EXP.relative_to(ROOT)),
        "baseline_local_estimate": baseline_score,
        "new_local_estimate": new_score,
        "delta": new_score - baseline_score,
        "gap_to_6500": 6500.0 - new_score,
        "gap_to_7000": 7000.0 - new_score,
        "gap_to_7400": 7400.0 - new_score,
        "gap_to_7600": 7600.0 - new_score,
        "improved_task_count": len(improved),
        "improved_tasks": sorted(improved),
        "improved_by_variant": dict(variant_counts),
        "candidate_status_counts": dict(status_counts),
        "zip_sanity": zip_sanity(OUTPUT_ZIP),
        "runtime_seconds": time.time() - started,
        "leakage_risk": "high: base exp016 includes signature lookup local-upper-bound candidates.",
        "overfitting_risk": "high until full arc-gen and private-like validation are run.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "exp016 bundleにも、strict bundleで効いた unused initializer prune の余地が残っている可能性がある。",
        "",
        "## Result",
        "",
        f"- baseline local estimate: `{baseline_score:.6f}`",
        f"- new local estimate: `{new_score:.6f}`",
        f"- delta: `{new_score - baseline_score:.6f}`",
        f"- gap to 6500: `{6500.0 - new_score:.6f}`",
        f"- improved tasks: `{sorted(improved)}`",
        "",
        "## Interpretation",
        "",
        "これはrule synthesisではなく、既存ONNXの低リスクなgraph surgeryである。改善が小さい場合でも、以降のDSL候補にも同じ後処理を適用する価値がある。",
        "",
        "## Risks",
        "",
        "- leakage risk: high。base は exp016 の signature lookup を含む。",
        "- overfitting risk: high。validation は sample20 であり、full arc-genではない。",
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp132_farm_step1_ir_emit_score.run_exp132 import build_programs
from experiments.neurogolf_farm import BundleCandidate, BundleLedger, CostExtractor, emit_candidate
from experiments.phase1_rewrite_utils import (
    evaluate_candidate,
    load_base_tasks,
    load_neurogolf_utils,
    point,
    validate_examples,
    write_zip,
    zip_sanity,
)


EXP_DIR = ROOT / "experiments" / "exp133_farm_step2_step3_validation_ledger"


def _float_or_zero(value: object) -> float:
    try:
        if value == "":
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    extractor = CostExtractor()
    rows: list[dict[str, object]] = []
    ledger_candidates: list[BundleCandidate] = []
    accepted_raws: dict[int, bytes] = {}

    for program in build_programs():
        task_id = int(program.task_id)
        base_task = base[task_id]
        candidate = emit_candidate(program)
        onnx_signal: dict[str, object] = {}
        if candidate.raw is not None:
            model_path = EXP_DIR / f"task{task_id:03d}_{candidate.template_name}.onnx"
            model = onnx.load_model_from_string(candidate.raw)
            try:
                model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
            except Exception:
                pass
            model_path.write_bytes(model.SerializeToString())
            onnx_signal = extractor.assess_onnx_path(model_path).as_dict()

        eval_row, sample_raw = evaluate_candidate(utils, candidate, base_task, arc_gen_sample=1, exp_dir=EXP_DIR)
        full_status = "not_run"
        full_reason = "sample validation failed"
        full_passed = 0
        full_failed = 0
        full_ok = False
        full_raw = sample_raw
        if sample_raw is not None:
            full_ok, full_reason, full_passed, full_failed = validate_examples(utils, sample_raw, task_id, arc_gen_sample=-1)
            full_status = f"{full_passed}_pass_{full_failed}_fail"

        candidate_cost = _float_or_zero(eval_row.candidate_cost)
        local_delta = point(candidate_cost) - base_task.points if candidate_cost > 0 else 0.0
        adoption_status = "accepted" if (
            full_ok and eval_row.status == "improved" and candidate_cost < base_task.cost
        ) else "rejected"
        risk = "low" if full_ok else "high"
        ledger_candidate = BundleCandidate(
            task_id=f"{task_id:03d}",
            source_exp="exp133_farm_step2_step3_validation_ledger",
            candidate_path=str(EXP_DIR / f"task{task_id:03d}_{candidate.template_name}.onnx") if candidate.raw else "",
            base_cost=float(base_task.cost),
            candidate_cost=candidate_cost,
            validation_status=full_status,
            local_delta=local_delta,
            risk=risk,
            adoption_status=adoption_status,
        )
        ledger_candidates.append(ledger_candidate)
        if ledger_candidate.accepted and full_raw is not None:
            accepted_raws[task_id] = full_raw

        rows.append(
            {
                "task_id": task_id,
                "family": program.family,
                "onnx_band": onnx_signal.get("predicted_cost_band", ""),
                "onnx_cost_proxy": onnx_signal.get("cost_proxy", ""),
                **asdict(eval_row),
                "sample_raw_available": sample_raw is not None,
                "full_validation_status": full_status,
                "full_validation_reason": full_reason,
                "full_ok": full_ok,
                "ledger_adoption_status": adoption_status,
                "ledger_risk": risk,
                "ledger_local_delta": local_delta,
                "ledger_accepted": ledger_candidate.accepted,
            }
        )

    ledger = BundleLedger(ledger_candidates)
    ledger_path = EXP_DIR / "bundle_ledger.csv"
    ledger.write_csv(ledger_path)

    bundle_info: dict[str, object] = {
        "bundle_created": False,
        "reason": "no accepted candidates",
    }
    if accepted_raws:
        bundle_raws = {task_id: task.raw for task_id, task in base.items()}
        bundle_raws.update(accepted_raws)
        bundle_path = EXP_DIR / "submission.zip"
        write_zip(bundle_path, bundle_raws)
        bundle_info = {
            "bundle_created": True,
            "submission_zip": str(bundle_path),
            "zip_sanity": zip_sanity(bundle_path),
            "accepted_task_count": len(accepted_raws),
        }

    rows_path = EXP_DIR / "step2_step3_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp": "exp133_farm_step2_step3_validation_ledger",
        "status": "pipeline_safe_operational",
        "scope": "Step2 full-arc validation plus Step3 bundle ledger and safe bundle gate",
        "n_candidates": len(rows),
        "n_full_ok": sum(1 for row in rows if row["full_ok"]),
        "n_accepted": len(ledger.accepted_candidates()),
        "total_local_delta": ledger.total_local_delta(),
        "bundle": bundle_info,
        "outputs": {
            "rows": str(rows_path),
            "bundle_ledger": str(ledger_path),
        },
        "rows": rows,
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Pipeline is safe-operational: it creates no submission bundle unless a candidate is full-arc valid and cost-improving.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(result)


def write_notes(result: dict[str, object]) -> None:
    lines = [
        "# exp133_farm_step2_step3_validation_ledger",
        "",
        "## Hypothesis",
        "",
        "Step 2/3としてfull-arc validationとBundleLedgerを接続すれば、採用候補がない場合は安全停止し、採用候補がある場合だけ差分bundleを作れる。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- n_candidates: `{result['n_candidates']}`",
        f"- n_full_ok: `{result['n_full_ok']}`",
        f"- n_accepted: `{result['n_accepted']}`",
        f"- total_local_delta: `{result['total_local_delta']}`",
        f"- bundle_created: `{result['bundle']['bundle_created']}`",
        "",
    ]
    for row in result["rows"]:
        lines.append(
            f"- task{row['task_id']:03d} {row['family']}: sample `{row['validation_status']}`, full `{row['full_validation_status']}`, adoption `{row['ledger_adoption_status']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Step 2/3の配管は稼働した。今回のcontrol候補は全てsample段階で不正解のため、full-arc validationへは進まず、ledgerも全件rejected。acceptedが0なのでsubmission.zipは生成しない。この安全停止が期待動作。",
            "",
            "## Next",
            "",
            "次はrule minerまたは既存rule-hit taskからplausible candidateを供給し、この同じpipelineでaccepted候補が出るか確認する。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "採用候補なし。local estimate/LBへ加算しない。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

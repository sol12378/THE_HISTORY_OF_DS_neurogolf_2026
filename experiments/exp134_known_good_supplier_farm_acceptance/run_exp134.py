from __future__ import annotations

import csv
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.neurogolf_farm import BundleCandidate, BundleLedger, CostExtractor
from experiments.phase1_rewrite_utils import load_neurogolf_utils, point, score_model, validate_examples


EXP_DIR = ROOT / "experiments" / "exp134_known_good_supplier_farm_acceptance"
SOURCE_EXP = ROOT / "experiments" / "exp127_focused_surgery_seventh_pass_limited"


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    source_result = json.loads((SOURCE_EXP / "result.json").read_text(encoding="utf-8"))
    accepted = source_result["accepted"]
    extractor = CostExtractor()
    rows = []
    ledger_candidates = []

    with zipfile.ZipFile(SOURCE_EXP / "submission.zip") as zf:
        for item in accepted:
            task_id = int(item["task_id"])
            raw = zf.read(f"task{task_id:03d}.onnx")
            model_path = EXP_DIR / f"task{task_id:03d}_{item['template_name']}.onnx"
            model_path.write_bytes(raw)
            signal = extractor.assess_onnx_path(model_path).as_dict()
            full_ok, full_reason, passed, failed = validate_examples(utils, raw, task_id, arc_gen_sample=-1)
            full_status = f"{passed}_pass_{failed}_fail"
            memory, params, score_reason = score_model(utils, raw, task_id, f"exp134_{item['template_name']}", EXP_DIR)
            official_cost = int(memory + params) if memory is not None and params is not None else None
            baseline_cost = float(item["baseline_cost"])
            candidate_cost = float(official_cost if official_cost is not None else item["candidate_cost"])
            local_delta = point(candidate_cost) - float(item["baseline_points"])
            adoption_status = "accepted" if (
                full_ok and official_cost is not None and candidate_cost < baseline_cost
            ) else "rejected"
            risk = "low" if full_ok else "high"
            ledger_candidate = BundleCandidate(
                task_id=f"{task_id:03d}",
                source_exp="exp134_known_good_supplier_farm_acceptance",
                candidate_path=str(model_path),
                base_cost=baseline_cost,
                candidate_cost=candidate_cost,
                validation_status=full_status,
                local_delta=local_delta,
                risk=risk,
                adoption_status=adoption_status,
            )
            ledger_candidates.append(ledger_candidate)
            rows.append(
                {
                    "task_id": task_id,
                    "template_name": item["template_name"],
                    "source_candidate_cost": item["candidate_cost"],
                    "official_cost": official_cost,
                    "score_reason": score_reason,
                    "full_validation_status": full_status,
                    "full_reason": full_reason,
                    "full_ok": full_ok,
                    "onnx_band": signal["predicted_cost_band"],
                    "onnx_cost_proxy": signal["cost_proxy"],
                    "ledger_adoption_status": adoption_status,
                    "ledger_accepted": ledger_candidate.accepted,
                    "local_delta": local_delta,
                }
            )

    ledger = BundleLedger(ledger_candidates)
    ledger_path = EXP_DIR / "bundle_ledger.csv"
    ledger.write_csv(ledger_path)
    rows_path = EXP_DIR / "known_good_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "exp": "exp134_known_good_supplier_farm_acceptance",
        "status": "known_good_supplier_complete",
        "source_exp": str(SOURCE_EXP),
        "n_candidates": len(rows),
        "n_full_ok": sum(1 for row in rows if row["full_ok"]),
        "n_accepted": len(ledger.accepted_candidates()),
        "total_local_delta": ledger.total_local_delta(),
        "rows": rows,
        "outputs": {"rows": str(rows_path), "bundle_ledger": str(ledger_path)},
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Known-good supplier can drive full-arc valid cost-improving candidates into BundleLedger accepted rows.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(result)


def write_notes(result: dict[str, object]) -> None:
    lines = [
        "# exp134_known_good_supplier_farm_acceptance",
        "",
        "## Hypothesis",
        "",
        "exp127のknown-good accepted候補をsupplierとしてfarmへ再投入すれば、full-arc validation/official score/BundleLedger acceptedまで通る。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- n_candidates: `{result['n_candidates']}`",
        f"- n_full_ok: `{result['n_full_ok']}`",
        f"- n_accepted: `{result['n_accepted']}`",
        f"- total_local_delta: `{result['total_local_delta']}`",
        "",
    ]
    for row in result["rows"]:
        lines.append(
            f"- task{row['task_id']:03d} {row['template_name']}: full `{row['full_validation_status']}`, official `{row['official_cost']}`, accepted `{row['ledger_accepted']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "control候補ではなくknown-good候補を入れると、farm ledgerがacceptedを出せることを確認した。これで候補supplierさえ強ければ、pipelineは採用・micro submission候補化まで進められる。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "exp127由来の既知full-validation gated graph surgery候補。新規LB加算ではなくpipeline acceptance再現実験。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

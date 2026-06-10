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

from experiments.neurogolf_farm import CostExtractor, IRNode, IRProgram, PrimitiveKind, TensorSpec, emit_candidate
from experiments.phase1_rewrite_utils import evaluate_candidate, load_base_tasks, load_neurogolf_utils


EXP_DIR = ROOT / "experiments" / "exp132_farm_step1_ir_emit_score"


def build_programs() -> list[IRProgram]:
    final = TensorSpec("output", (1, 10, 30, 30))
    patch = TensorSpec("crop", (1, 10, 3, 3))
    programs: list[IRProgram] = []
    for task_id in [1, 2, 3, 4, 5]:
        programs.append(
            IRProgram(
                task_id=str(task_id),
                family="identity",
                intent="Step1 real-task identity control",
                nodes=(IRNode("identity", PrimitiveKind.ONE_NODE_DATA_MOVEMENT, "Identity", output=final),),
                source="exp132_step1",
            )
        )
    programs.append(
        IRProgram(
            task_id="1",
            family="channel_gather",
            intent="Step1 real-task channel gather control",
            nodes=(IRNode("channel_gather", PrimitiveKind.CHANNEL_GATHER, "Gather", output=final, attrs={"order": list(range(10))}),),
            source="exp132_step1",
        )
    )
    programs.append(
        IRProgram(
            task_id="1",
            family="static_slice_pad",
            intent="Step1 real-task static crop cost control",
            nodes=(
                IRNode(
                    "slice_pad",
                    PrimitiveKind.STATIC_SLICE_PAD,
                    "SlicePad",
                    output=patch,
                    attrs={
                        "starts": [0, 0, 0, 0],
                        "ends": [1, 10, 3, 3],
                        "axes": [0, 1, 2, 3],
                        "pads": [0, 0, 0, 0, 0, 0, 27, 27],
                    },
                ),
            ),
            source="exp132_step1",
        )
    )
    return programs


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    utils = load_neurogolf_utils()
    base = load_base_tasks()
    extractor = CostExtractor()
    rows = []
    for program in build_programs():
        task_id = int(program.task_id)
        ir_signal = extractor.assess_ir(program).as_dict()
        candidate = emit_candidate(program)
        onnx_signal = {}
        if candidate.raw is not None:
            model_path = EXP_DIR / f"task{task_id:03d}_{candidate.template_name}.onnx"
            model = onnx.load_model_from_string(candidate.raw)
            try:
                model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
            except Exception:
                pass
            model_path.write_bytes(model.SerializeToString())
            onnx_signal = extractor.assess_onnx_path(model_path).as_dict()
        eval_row, accepted_raw = evaluate_candidate(utils, candidate, base[task_id], arc_gen_sample=1, exp_dir=EXP_DIR)
        row = {
            "task_id": task_id,
            "family": program.family,
            "candidate_status": candidate.status,
            "candidate_reason": candidate.reason,
            "ir_band": ir_signal["predicted_cost_band"],
            "ir_cost_proxy": ir_signal["cost_proxy"],
            "onnx_band": onnx_signal.get("predicted_cost_band", ""),
            "onnx_cost_proxy": onnx_signal.get("cost_proxy", ""),
            **asdict(eval_row),
            "accepted_raw": accepted_raw is not None,
        }
        rows.append(row)

    csv_path = EXP_DIR / "step1_rows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    improved = [row for row in rows if row["status"] == "improved"]
    result = {
        "exp": "exp132_farm_step1_ir_emit_score",
        "status": "step1_complete",
        "scope": "IRProgram -> ONNX emitter -> official score/evaluate_candidate on real tasks; arc_gen_sample=1 only",
        "n_programs": len(rows),
        "n_improved": len(improved),
        "rows": rows,
        "outputs": {"step1_rows": str(csv_path)},
        "elapsed_s": round(time.time() - t0, 3),
        "next_step": "Step2 connects full-arc validation and converts rows into BundleCandidate ledger entries.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_notes(result)


def write_notes(result: dict[str, object]) -> None:
    lines = [
        "# exp132_farm_step1_ir_emit_score",
        "",
        "## Hypothesis",
        "",
        "実taskに対してIRProgram -> ONNX emitter -> official score/evaluate_candidateを接続すれば、farm OSの最初の実パイプラインとして進捗確認できる。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- n_programs: `{result['n_programs']}`",
        f"- n_improved: `{result['n_improved']}`",
        "",
    ]
    for row in result["rows"]:
        lines.append(
            f"- task{row['task_id']:03d} {row['family']}: status `{row['status']}`, validation `{row['validation_status']}`, cost `{row['candidate_cost']}`, proxy `{row['onnx_cost_proxy']}`"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Step 1は接続性確認であり、score-producing候補の採用はまだ行わない。identity/channel_gather/static_slice_padが実taskの既存評価関数へ流れ、validationとofficial score結果がCSV化されれば成功。",
            "",
            "## Next",
            "",
            "Step 2でfull-arc validationを明示接続し、BundleCandidateへ変換する。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

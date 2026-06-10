"""Compiler farm orchestration primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .cost_extractor import CostExtractor
from .bundle_manager import BundleCandidate, BundleLedger
from .ir import IRNode, IRProgram, PrimitiveKind, TensorSpec, gridsample_program, recolor_cast_program, recolor_direct_program
from .public_code import PublicCodeRegistry


@dataclass(frozen=True)
class FarmConfig:
    workspace: Path
    experiment_dir: Path
    floor_target_lb: float = 6285.0
    submitted_best_estimate: float = 6008.96


class FarmRunner:
    def __init__(self, config: FarmConfig) -> None:
        self.config = config
        self.extractor = CostExtractor()

    def run_smoke(self) -> dict[str, object]:
        self.config.experiment_dir.mkdir(parents=True, exist_ok=True)
        programs = self._smoke_programs()
        guardrail_rows = [self.extractor.assess_ir(program).as_dict() for program in programs]
        registry = PublicCodeRegistry.from_experiment_results(self.config.workspace)
        registry_path = self.config.experiment_dir / "public_code_registry.csv"
        registry.write_csv(registry_path)
        ledger = BundleLedger(
            [
                BundleCandidate(
                    task_id="smoke_accept_low_cost",
                    source_exp="exp129",
                    candidate_path="",
                    base_cost=1000.0,
                    candidate_cost=500.0,
                    validation_status="266_pass_0_fail",
                    local_delta=0.1,
                    risk="low",
                    adoption_status="accepted",
                ),
                BundleCandidate(
                    task_id="smoke_reject_failed_validation",
                    source_exp="exp129",
                    candidate_path="",
                    base_cost=1000.0,
                    candidate_cost=100.0,
                    validation_status="24_pass_1_fail",
                    local_delta=0.5,
                    risk="low",
                    adoption_status="accepted",
                ),
            ]
        )
        ledger_path = self.config.experiment_dir / "bundle_ledger_smoke.csv"
        ledger.write_csv(ledger_path)
        accepted_candidates = ledger.accepted_candidates()
        best_candidates_by_task = ledger.best_candidates_by_task()
        low_cost_min_by_primitive: dict[str, int] = {}
        for row in guardrail_rows:
            if row["predicted_cost_band"] != "250-600_plausible" or row["cost_proxy"] is None:
                continue
            for primitive_kind in row.get("primitive_kinds", ()):
                current = low_cost_min_by_primitive.get(str(primitive_kind))
                cost_proxy = int(row["cost_proxy"])
                if current is None or cost_proxy < current:
                    low_cost_min_by_primitive[str(primitive_kind)] = cost_proxy

        result = {
            "exp_id": self.config.experiment_dir.name,
            "status": "pipeline_core_ready",
            "floor_status": registry.floor_status(self.config.floor_target_lb),
            "guardrail_rows": guardrail_rows,
            "low_cost_guardrail_subjects": [
                row["subject"] for row in guardrail_rows if row["predicted_cost_band"] == "250-600_plausible"
            ],
            "low_cost_guardrail_min_cost_by_primitive": low_cost_min_by_primitive,
            "outputs": {
                "public_code_registry": str(registry_path),
                "bundle_ledger_smoke": str(ledger_path),
            },
            "bundle_smoke": {
                "accepted_count": len(accepted_candidates),
                "best_by_task_count": len(best_candidates_by_task),
                "duplicate_task_count": len(accepted_candidates) - len(best_candidates_by_task),
                "total_local_delta": ledger.total_local_delta(),
                "best_total_local_delta": ledger.best_total_local_delta(),
                "submission_decision": ledger.submission_decision(self.config.submitted_best_estimate),
                "submission_decision_scope": "smoke_dummy_not_kaggle_candidate",
                "submission_decision_reason": "run_smoke uses synthetic ledger rows only; do not submit from this decision",
            },
            "decision": (
                "ONNX farm should accept only low-cost primitive IR or full-arc-safe graph surgery. "
                "Public CODE is imported as teacher/intelligence until it has full validation and LB evidence."
            ),
        }
        (self.config.experiment_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    @staticmethod
    def _smoke_programs() -> list[IRProgram]:
        final_grid = TensorSpec("output", (1, 10, 30, 30))
        small_patch = TensorSpec("patch", (1, 10, 6, 6))
        return [
            recolor_direct_program("smoke_recolor_direct", output_shape=(1, 1, 1, 1), source="exp285 factory"),
            recolor_cast_program("smoke_recolor_cast", output_shape=(1, 1, 1, 1), source="exp285 factory"),
            gridsample_program("smoke_gridsample", output_shape=(1, 1, 1, 1), source="exp301 factory"),
            IRProgram(
                task_id="smoke_channel_gather",
                family="one_node_colormap",
                intent="verify cheap fused channel recolor lane",
                nodes=(
                    IRNode(
                        name="channel_gather",
                        kind=PrimitiveKind.CHANNEL_GATHER,
                        op_type="Gather",
                        output=final_grid,
                    ),
                ),
                source="exp094/exp100 archetype",
            ),
            IRProgram(
                task_id="smoke_slice_pad",
                family="small_crop",
                intent="verify small Slice+Pad lane",
                nodes=(
                    IRNode("slice", PrimitiveKind.STATIC_SLICE_PAD, "Slice", output=small_patch),
                    IRNode("pad", PrimitiveKind.STATIC_SLICE_PAD, "Pad", output=final_grid),
                ),
                source="exp086/exp100 archetype",
            ),
            IRProgram(
                task_id="smoke_full_grid_where",
                family="bad_full_grid_overlay",
                intent="reject task366-style full-grid overlay",
                nodes=(
                    IRNode("mask", PrimitiveKind.FULL_GRID_COMPOSITION, "Where", output=final_grid),
                    IRNode("overlay", PrimitiveKind.FULL_GRID_COMPOSITION, "Where", output=final_grid),
                ),
                source="exp082 negative control",
            ),
            IRProgram(
                task_id="smoke_scatter_writeback",
                family="bad_sparse_writeback",
                intent="reject ScatterND sparse writeback",
                nodes=(
                    IRNode("scatter", PrimitiveKind.SPARSE_WRITEBACK, "ScatterND", output=final_grid),
                ),
                source="exp083 negative control",
            ),
        ]

    @staticmethod
    def write_notes(path: str | Path, result: dict[str, object]) -> None:
        out = Path(path)
        floor = result["floor_status"]
        submission_decision = result["bundle_smoke"].get("submission_decision", {})
        out.write_text(
            "\n".join(
                [
                    "# exp129_compiler_farm_core",
                    "",
                    "## Hypothesis",
                    "",
                    "NeuroGolf専用IRとcost extractorをONNX生成前に置けば、`params + output tensor bytes` のlocal estimateで候補を事前評価でき、実験farmをscore-producing候補へ集中できる。",
                    "",
                    "## Result",
                    "",
                    f"- status: `{result['status']}`",
                    f"- 6285 floor ready: `{floor['submit_floor_ready']}`",
                    f"- max observed Kaggle LB in registry: `{floor['max_observed_kaggle_lb']}`",
                    "- public CODEはregistryへ取り込んだが、6285 submit floorとしては未証明。full-arc validationとLB evidenceがあるものだけfloorに昇格する。",
                    "- smoke guardrailではcandidate costをoutput tensor bytes中心に表示し、`Where` はMAC=0として扱う。`FULL_GRID_COMPOSITION` / `ScatterND` などの構造リスクは別guardrailで維持する。",
                    f"- bundle ledger smoke accepted count: `{result['bundle_smoke']['accepted_count']}`",
                    f"- bundle ledger smoke best-by-task count: `{result['bundle_smoke'].get('best_by_task_count')}`",
                    f"- bundle ledger smoke duplicate task count: `{result['bundle_smoke'].get('duplicate_task_count')}`",
                    f"- submit-gate local delta uses best-total value: `{result['bundle_smoke'].get('best_total_local_delta')}`",
                    f"- submitted best estimate: `{submission_decision.get('submitted_best_estimate')}`",
                    f"- candidate estimate: `{submission_decision.get('candidate_estimate')}`",
                    f"- should submit: `{submission_decision.get('should_submit')}`",
                    f"- submission decision scope: `{result['bundle_smoke'].get('submission_decision_scope')}`",
                    f"- submission decision reason: `{result['bundle_smoke'].get('submission_decision_reason')}`",
                    "",
                    "## Leakage / Overfitting Risk",
                    "",
                    "公開CODEやblend artifactを直接best bundleへ加算するとpublic LB overfitとvalidation leakageのリスクがある。今回のpipelineではteacher/intelligenceとして登録し、公式互換validationとKaggle較正が揃うまで提出下限とはみなさない。",
                    "",
                    "## Next",
                    "",
                    "1. Kaggle public CODE notebook URL/sourceをregistryに追加し、claimed LB 6285候補を個別にfull-arc再検証する。",
                    "2. IR lowering pluginを `phase1_rewrite_utils.py` のcandidate evaluationへ接続する。",
                    "3. accepted candidate bundleにSeddik-style post-passとfocused surgeryを標準適用する。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

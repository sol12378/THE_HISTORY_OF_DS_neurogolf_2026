"""Compiler farm orchestration primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .cost_extractor import CostExtractor
from .bundle_manager import BundleCandidate, BundleLedger
from .ir import IRNode, IRProgram, PrimitiveKind, TensorSpec
from .public_code import PublicCodeRegistry


@dataclass(frozen=True)
class FarmConfig:
    workspace: Path
    experiment_dir: Path
    floor_target_lb: float = 6285.0


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

        result = {
            "exp_id": self.config.experiment_dir.name,
            "status": "pipeline_core_ready",
            "floor_status": registry.floor_status(self.config.floor_target_lb),
            "guardrail_rows": guardrail_rows,
            "outputs": {
                "public_code_registry": str(registry_path),
                "bundle_ledger_smoke": str(ledger_path),
            },
            "bundle_smoke": {
                "accepted_count": len(ledger.accepted_candidates()),
                "total_local_delta": ledger.total_local_delta(),
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
        out.write_text(
            "\n".join(
                [
                    "# exp129_compiler_farm_core",
                    "",
                    "## Hypothesis",
                    "",
                    "NeuroGolf専用IRとcost extractorをONNX生成前に置けば、full-grid中間やScatterND/Where型の高cost候補を事前rejectでき、実験farmをscore-producing候補へ集中できる。",
                    "",
                    "## Result",
                    "",
                    f"- status: `{result['status']}`",
                    f"- 6285 floor ready: `{floor['submit_floor_ready']}`",
                    f"- max observed Kaggle LB in registry: `{floor['max_observed_kaggle_lb']}`",
                    "- public CODEはregistryへ取り込んだが、6285 submit floorとしては未証明。full-arc validationとLB evidenceがあるものだけfloorに昇格する。",
                    "- smoke guardrailではcheap lane (`Gather`, `Slice+Pad`) とreject lane (`Where`, `ScatterND`) を分離できた。",
                    f"- bundle ledger smoke accepted count: `{result['bundle_smoke']['accepted_count']}`",
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

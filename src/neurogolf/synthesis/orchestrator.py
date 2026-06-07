from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PHASE_TARGETS: dict[str, tuple[int, int]] = {
    "phase1_6500_bridge": (100, 9000),
    "phase2_7000_submit_safe": (200, 3000),
    "phase3_7400_private_like": (320, 1000),
    "phase4_7600_contender": (400, 300),
}


@dataclass(frozen=True)
class FamilyStrategy:
    family: str
    system: str
    first_actions: tuple[str, ...]
    preferred_lowerings: tuple[str, ...]
    reject_lowerings: tuple[str, ...]
    proof_requirements: tuple[str, ...]
    leakage_risk: str
    private_risk: str


@dataclass(frozen=True)
class SynthesisQueueItem:
    queue_rank: int
    task_id: int
    target_phase: str
    synthesis_family: str
    route_prediction: str
    current_source: str
    current_template: str
    current_cost: int
    current_points: float
    target_cost: int
    projected_gain_to_target: float
    priority_score: float
    required_system: str
    first_action: str
    preferred_lowerings: str
    reject_lowerings: str
    proof_requirements: str
    worker_acceptance: str
    leakage_risk: str
    private_risk: str


@dataclass(frozen=True)
class LoweringGuardrail:
    pattern: str
    decision: str
    max_cost_ratio: float | str
    banned_ops: str
    rationale: str
    replacement: str


FAMILY_STRATEGIES: dict[str, FamilyStrategy] = {
    "signature_lookup_current": FamilyStrategy(
        family="signature_lookup_current",
        system="lookup compression and rule extraction",
        first_actions=(
            "mine train/test signatures for invariant features",
            "fit a shallow decision tree or affine hash over colors/shapes/anchors",
            "attempt DSL replacement before another lookup variant",
        ),
        preferred_lowerings=(
            "small Gather over learned color roles",
            "constant Slice/Gather for extracted crop cases",
            "initializer-pruned decision tree",
        ),
        reject_lowerings=(
            "large per-example ScatterND coordinate table",
            "full arc-gen memorization",
            "dynamic hash table with large MatMul",
        ),
        proof_requirements=(
            "all train/test examples pass",
            "arc-gen sample pass",
            "rule description does not mention hidden labels",
            "cost is lower than current selected ONNX",
        ),
        leakage_risk="high",
        private_risk="high until replaced by rule",
    ),
    "crop_resize": FamilyStrategy(
        family="crop_resize",
        system="object-anchor crop/resize synthesizer",
        first_actions=(
            "enumerate fixed crop, object-centered crop, mask-color crop",
            "fit output shape rule from train examples",
            "prefer constant Slice before dynamic bbox",
        ),
        preferred_lowerings=(
            "Slice with constant starts/ends",
            "Gather over static row/column indices",
            "small Pad/Concat only when shape is fixed",
        ),
        reject_lowerings=(
            "NonZero or Compress bbox extraction",
            "full-grid Tile plus mask",
            "large ScatterND for every output cell",
        ),
        proof_requirements=(
            "shape rule fits every train pair",
            "sample validation pass",
            "no dynamic shape tensors",
            "cost target is met before full validation",
        ),
        leakage_risk="medium",
        private_risk="medium",
    ),
    "sparse_edit_or_object_completion": FamilyStrategy(
        family="sparse_edit_or_object_completion",
        system="sparse edit and object completion enumerator",
        first_actions=(
            "derive changed-cell mask and color roles",
            "fit local neighborhood, component, or ray rule",
            "emit sparse coordinate program only if index table is tiny",
        ),
        preferred_lowerings=(
            "small Conv kernels",
            "GatherND/ScatterND with few coordinates",
            "Equal plus single Where for small masks",
        ),
        reject_lowerings=(
            "full-grid Where chain",
            "large coordinate initializer",
            "unrolled iterative fill without cost forecast",
        ),
        proof_requirements=(
            "changed-cell rule explains all train diffs",
            "arc-gen sample pass",
            "coordinate table remains below gate",
            "cost is lower than current selected ONNX",
        ),
        leakage_risk="medium",
        private_risk="medium",
    ),
    "point_to_line_pattern": FamilyStrategy(
        family="point_to_line_pattern",
        system="seed-to-pattern grammar",
        first_actions=(
            "identify seed colors and directions",
            "enumerate rays, periodic diagonal shifts, boxes, and reflected motifs",
            "bind colors through roles rather than absolute labels when possible",
        ),
        preferred_lowerings=(
            "constant affine coordinate masks",
            "small Conv kernels for line detection",
            "Gather/Where over precomputed row-column masks",
        ),
        reject_lowerings=(
            "per-cell ScatterND table",
            "full-grid iterative ray propagation",
            "Where chain over all periods",
        ),
        proof_requirements=(
            "direction/period inferred from train examples",
            "sample validation pass",
            "private-like rule note names the invariant",
            "cost is lower than current selected ONNX",
        ),
        leakage_risk="medium",
        private_risk="medium",
    ),
    "region_partition_fill": FamilyStrategy(
        family="region_partition_fill",
        system="region and room-fill synthesizer",
        first_actions=(
            "detect boundary colors and line-grid rooms",
            "prefer closed-form room masks over flood-fill unroll",
            "try row/column prefix reductions before dilation",
        ),
        preferred_lowerings=(
            "static row/column ReduceSum masks",
            "small Conv kernels",
            "single-pass room mask Where",
        ),
        reject_lowerings=(
            "radius-scale flood-fill unroll",
            "many-step dilation",
            "dynamic full-grid color map",
        ),
        proof_requirements=(
            "boundary/interior semantics fit train examples",
            "sample validation pass",
            "unroll count is justified by cost gate",
            "cost is lower than current selected ONNX",
        ),
        leakage_risk="medium",
        private_risk="medium-high",
    ),
    "line_grid_fill": FamilyStrategy(
        family="line_grid_fill",
        system="line, ray, and grid grammar",
        first_actions=(
            "detect complete/incomplete rows and columns",
            "enumerate line extension and crossing rules",
            "fit stop conditions from blockers and colors",
        ),
        preferred_lowerings=(
            "ReduceSum row/column masks",
            "small Conv kernels",
            "constant masks plus Where",
        ),
        reject_lowerings=(
            "per-cell ScatterND table",
            "full-grid Tile",
            "long Where chain",
        ),
        proof_requirements=(
            "line stop condition is explicit",
            "sample validation pass",
            "static shape check pass",
            "cost is lower than current selected ONNX",
        ),
        leakage_risk="medium",
        private_risk="medium",
    ),
    "color_map": FamilyStrategy(
        family="color_map",
        system="color role normalizer",
        first_actions=(
            "fit consistent color map",
            "check whether color map is a substep for another family",
            "emit only if it beats current graph",
        ),
        preferred_lowerings=("1x1 Conv", "small Equal/Where chain"),
        reject_lowerings=("large MatMul color map", "dynamic full-grid color map"),
        proof_requirements=("consistent color mapping", "sample validation pass", "cost is lower than current selected ONNX"),
        leakage_risk="low",
        private_risk="medium",
    ),
    "same_shape_global_transform": FamilyStrategy(
        family="same_shape_global_transform",
        system="global transform and artifact surgery",
        first_actions=(
            "test flip/rotate/transpose/color-map sketches",
            "inspect current artifact for unused initializers",
            "prefer graph surgery when artifact is already compact",
        ),
        preferred_lowerings=("Transpose", "Gather", "initializer prune", "1x1 Conv"),
        reject_lowerings=("rewriting compact artifact with larger full-grid graph",),
        proof_requirements=("transform proof fits train examples", "sample validation pass", "cost is lower than current selected ONNX"),
        leakage_risk="low",
        private_risk="medium",
    ),
}


LOWERING_GUARDRAILS: tuple[LoweringGuardrail, ...] = (
    LoweringGuardrail(
        pattern="full_grid_where_chain",
        decision="pre_reject_when_grid_wide_or_repeated",
        max_cost_ratio=0.85,
        banned_ops="",
        rationale="Past correct rules became more expensive when every cell passed through repeated Where nodes.",
        replacement="derive a compact mask with Conv/Reduce or use constant Slice/Gather.",
    ),
    LoweringGuardrail(
        pattern="large_scatternd_coordinate_initializer",
        decision="pre_reject_when_coordinate_count_is_not_tiny",
        max_cost_ratio=0.75,
        banned_ops="",
        rationale="ScatterND can be valid but loses if coordinates encode most output cells.",
        replacement="compress to affine coordinates, constant masks, or a smaller decision tree.",
    ),
    LoweringGuardrail(
        pattern="full_grid_tile_plus_mask",
        decision="pre_reject_for_crop_resize_unless_tiled_area_is_small",
        max_cost_ratio=0.70,
        banned_ops="",
        rationale="Tile inflates activation memory under the official-like score.",
        replacement="Slice/Gather/Concat with static output placement.",
    ),
    LoweringGuardrail(
        pattern="dynamic_full_grid_colormap",
        decision="pre_reject_when_color_map_depends_on_large_matmul",
        max_cost_ratio=0.80,
        banned_ops="",
        rationale="Dynamic MatMul color maps validated but cost more than compact public artifacts.",
        replacement="small 1x1 Conv, Equal/Where, or color-role constants.",
    ),
    LoweringGuardrail(
        pattern="forbidden_dynamic_shape_ops",
        decision="always_reject",
        max_cost_ratio="n/a",
        banned_ops="Compress, NonZero, Loop, Scan, Unique, Function, Script, Sequence, subgraph",
        rationale="These are outside the strict submission envelope used by the project.",
        replacement="static-shape masks, constant indices, and bounded primitive sketches.",
    ),
)


def point(cost: int | float) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def gain_to_cost(current_cost: int, target_cost: int) -> float:
    if current_cost <= target_cost:
        return 0.0
    return point(target_cost) - point(current_cost)


def phase_for_rank(rank: int) -> str:
    for phase, (scope, _) in PHASE_TARGETS.items():
        if rank <= scope:
            return phase
    return "phase4_7600_contender"


def strategy_for(family: str) -> FamilyStrategy:
    return FAMILY_STRATEGIES.get(
        family,
        FamilyStrategy(
            family=family,
            system="generic DSL enumerator",
            first_actions=("inspect train examples", "fit a small static-shape sketch"),
            preferred_lowerings=("static primitive composition",),
            reject_lowerings=("dynamic shape", "large full-grid tables"),
            proof_requirements=("train/test pass", "sample validation pass", "cost is lower than current selected ONNX"),
            leakage_risk="unknown",
            private_risk="unknown",
        ),
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dataclass_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def build_queue(backlog_rows: list[dict[str, str]], selected_rows: list[dict[str, str]]) -> list[SynthesisQueueItem]:
    selected_by_task = {int(row["task_id"]): row for row in selected_rows}
    backlog_by_task = {int(row["task_id"]): row for row in backlog_rows}
    current_rows: list[dict[str, Any]] = []
    for task_id, selected in selected_by_task.items():
        backlog = backlog_by_task.get(task_id, {})
        cost = int(float(selected["cost"]))
        current_rows.append(
            {
                "task_id": task_id,
                "cost": cost,
                "points": float(selected.get("local_points") or point(cost)),
                "source": selected.get("source", ""),
                "template": selected.get("template_name", ""),
                "route": selected.get("route", backlog.get("route_prediction", "")),
                "family": backlog.get("synthesis_family", "unknown_program_synthesis"),
            }
        )
    current_rows.sort(key=lambda row: (-int(row["cost"]), int(row["task_id"])))

    items: list[SynthesisQueueItem] = []
    for cost_rank, row in enumerate(current_rows, start=1):
        phase = phase_for_rank(cost_rank)
        _, target_cost = PHASE_TARGETS[phase]
        family = str(row["family"])
        strategy = strategy_for(family)
        gain = gain_to_cost(int(row["cost"]), target_cost)
        risk_bonus = 0.35 if family == "signature_lookup_current" else 0.10 if "lookup" in str(row["template"]) else 0.0
        phase_bonus = {"phase1_6500_bridge": 2.0, "phase2_7000_submit_safe": 1.0, "phase3_7400_private_like": 0.4}.get(phase, 0.0)
        priority_score = gain * 10.0 + math.log(max(1, int(row["cost"]))) + risk_bonus + phase_bonus
        items.append(
            SynthesisQueueItem(
                queue_rank=0,
                task_id=int(row["task_id"]),
                target_phase=phase,
                synthesis_family=family,
                route_prediction=str(row["route"]),
                current_source=str(row["source"]),
                current_template=str(row["template"]),
                current_cost=int(row["cost"]),
                current_points=float(row["points"]),
                target_cost=target_cost,
                projected_gain_to_target=gain,
                priority_score=priority_score,
                required_system=strategy.system,
                first_action="; ".join(strategy.first_actions),
                preferred_lowerings="; ".join(strategy.preferred_lowerings),
                reject_lowerings="; ".join(strategy.reject_lowerings),
                proof_requirements="; ".join(strategy.proof_requirements),
                worker_acceptance=(
                    f"task{int(row['task_id']):03d}: validation pass, cost < {int(row['cost'])}, "
                    f"target cost <= {target_cost}, static/banned-op gate pass"
                ),
                leakage_risk=strategy.leakage_risk,
                private_risk=strategy.private_risk,
            )
        )

    phase_order = {
        "phase1_6500_bridge": 0,
        "phase2_7000_submit_safe": 1,
        "phase3_7400_private_like": 2,
        "phase4_7600_contender": 3,
    }
    items.sort(key=lambda item: (phase_order.get(item.target_phase, 9), -item.priority_score, item.task_id))
    return [
        SynthesisQueueItem(**{**asdict(item), "queue_rank": idx})
        for idx, item in enumerate(items, start=1)
    ]


def summarize_queue(items: list[SynthesisQueueItem]) -> dict[str, Any]:
    phase_counts = Counter(item.target_phase for item in items)
    family_counts = Counter(item.synthesis_family for item in items)
    system_counts = Counter(item.required_system for item in items)
    projected: dict[str, Any] = {}
    by_cost = sorted(items, key=lambda item: (-item.current_cost, item.task_id))
    for phase, (scope, target_cost) in PHASE_TARGETS.items():
        gain = sum(gain_to_cost(item.current_cost, target_cost) for item in by_cost[:scope])
        projected[phase] = {
            "scope_top_n": scope,
            "target_cost": target_cost,
            "projected_gain": gain,
        }
    return {
        "task_count": len(items),
        "phase_counts": dict(sorted(phase_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "system_counts": dict(system_counts.most_common()),
        "projected": projected,
        "top_queue": [asdict(item) for item in items[:25]],
    }


def program_schema() -> dict[str, Any]:
    return {
        "program_candidate": {
            "task_id": "int",
            "candidate_id": "stable string",
            "family": "one synthesis family",
            "program_steps": [{"primitive": "string", "params": "json object"}],
            "lowering_plan": {
                "onnx_ops": ["static ONNX op names"],
                "expected_cost_upper_bound": "int",
                "expected_file_bytes_upper_bound": "int",
                "banned_ops": ["must be empty before emission"],
            },
            "validation_plan": {
                "train": "required",
                "test": "required",
                "arc_gen_sample": "default 20, full for submit candidate",
                "private_like_family_holdout": "required for phase3+",
            },
            "proof_log": {
                "invariant": "human-readable rule invariant",
                "fit_evidence": "which examples support the rule",
                "failure_modes": "known cases to reject",
                "leakage_risk": "low/medium/high",
                "overfitting_risk": "low/medium/high",
            },
        }
    }


def guardrail_rows() -> list[dict[str, Any]]:
    return dataclass_rows(LOWERING_GUARDRAILS)

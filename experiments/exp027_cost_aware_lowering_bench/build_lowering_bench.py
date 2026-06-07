from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import date
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurogolf.synthesis.orchestrator import guardrail_rows, load_csv, write_csv  # noqa: E402


EXP_ID = "exp027_cost_aware_lowering_bench"
EXP_DIR = ROOT / "experiments" / EXP_ID

SOURCE_MANIFESTS = [
    ROOT / "experiments" / "exp016_top100_rewrite_campaign" / "candidate_manifest.csv",
    ROOT / "experiments" / "exp019_boundary_flood_fill_campaign" / "candidate_manifest.csv",
    ROOT / "experiments" / "exp020_diagonal_periodic_lowmem" / "candidate_manifest.csv",
    ROOT / "experiments" / "exp021_diagonal_scatternd_lowmem" / "candidate_manifest.csv",
    ROOT / "experiments" / "exp024_ring_depth_dynamic_colormap" / "candidate_manifest.csv",
    ROOT / "experiments" / "exp025_tile_prefix_blocks_task221" / "candidate_manifest.csv",
]

BASE_RESULT_PATH = ROOT / "experiments" / "exp023_graph_surgery_exp016" / "result.json"


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lowering_pattern(template_name: str) -> str:
    name = template_name.lower()
    if "ring_depth" in name:
        return "dynamic_full_grid_colormap"
    if "periodic_shift" in name:
        return "full_grid_where_chain"
    if "diagonal_shift" in name and "scatternd" in name:
        return "large_scatternd_coordinate_initializer"
    if "tile_prefix_blocks_sparse" in name:
        return "dynamic_argmax_scatternd"
    if "tile_prefix_blocks" in name:
        return "full_grid_tile_plus_mask"
    if "signature" in name and "scatter" in name:
        return "memorized_signature_scatternd"
    if "boundary_flood" in name:
        return "unrolled_boundary_flood_fill"
    if "crop" in name:
        return "constant_slice_crop_candidate"
    if "conv_color_map" in name or "global_transform_color_map" in name:
        return "small_color_or_global_transform_candidate"
    if "identity" in name:
        return "identity_sentinel"
    return "other_template_candidate"


def decision_for(pattern: str, status: str, ratio: float | None) -> str:
    if pattern in {
        "dynamic_full_grid_colormap",
        "full_grid_where_chain",
        "large_scatternd_coordinate_initializer",
        "dynamic_argmax_scatternd",
        "full_grid_tile_plus_mask",
        "unrolled_boundary_flood_fill",
    }:
        return "pre_reject_unless_static_cost_forecast_beats_baseline"
    if pattern == "memorized_signature_scatternd":
        return "allow_only_as_local_upper_bound_or_if_compressed"
    if pattern in {"constant_slice_crop_candidate", "small_color_or_global_transform_candidate"}:
        return "allow_when_validation_and_cost_pass"
    if status == "improved":
        return "allow_and_record_as_known_good"
    if ratio is not None and ratio >= 1.0:
        return "reject_after_cost_check"
    return "inspect_manually"


def read_observations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest in SOURCE_MANIFESTS:
        if not manifest.exists():
            continue
        exp_id = manifest.parent.name
        for row in load_csv(manifest):
            candidate_cost = row.get("candidate_cost", "")
            baseline_cost = row.get("baseline_cost", "")
            ratio: float | None = None
            point_delta: float | None = None
            if candidate_cost and baseline_cost:
                try:
                    ratio = float(candidate_cost) / max(1.0, float(baseline_cost))
                    cp = row.get("candidate_points", "")
                    bp = row.get("baseline_points", "")
                    if cp and bp:
                        point_delta = float(cp) - float(bp)
                except ValueError:
                    ratio = None
            pattern = lowering_pattern(row.get("template_name", ""))
            rows.append(
                {
                    "exp_id": exp_id,
                    "task_id": row.get("task_id", ""),
                    "template_name": row.get("template_name", ""),
                    "pattern": pattern,
                    "status": row.get("status", ""),
                    "validation_status": row.get("validation_status", ""),
                    "baseline_cost": baseline_cost,
                    "candidate_cost": candidate_cost,
                    "cost_ratio": f"{ratio:.6f}" if ratio is not None else "",
                    "point_delta": f"{point_delta:.6f}" if point_delta is not None else "",
                    "reason": row.get("reason", ""),
                    "decision": decision_for(pattern, row.get("status", ""), ratio),
                }
            )
    return rows


def summarize_patterns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pattern"])].append(row)

    out: list[dict[str, Any]] = []
    for pattern, items in sorted(grouped.items()):
        numeric_ratios = [float(row["cost_ratio"]) for row in items if row["cost_ratio"]]
        pass_count = sum(1 for row in items if str(row["validation_status"]).endswith("_0_fail"))
        improved = sum(1 for row in items if row["status"] == "improved")
        no_gain = sum(1 for row in items if row["status"] == "no_cost_gain")
        skipped = sum(1 for row in items if row["status"] == "skipped")
        decisions = Counter(row["decision"] for row in items)
        out.append(
            {
                "pattern": pattern,
                "observations": len(items),
                "validation_pass_observations": pass_count,
                "improved": improved,
                "no_cost_gain": no_gain,
                "skipped": skipped,
                "max_cost_ratio": f"{max(numeric_ratios):.6f}" if numeric_ratios else "",
                "min_cost_ratio": f"{min(numeric_ratios):.6f}" if numeric_ratios else "",
                "avg_cost_ratio": f"{sum(numeric_ratios) / len(numeric_ratios):.6f}" if numeric_ratios else "",
                "decision": decisions.most_common(1)[0][0],
            }
        )
    return out


def notes_text(result: dict[str, Any]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "7600向けprogram synthesisでは、候補ONNXを大量に生成する前にcost-aware gateを置く必要がある。"
        "過去の失敗候補を集約すれば、validation passしても重くなるloweringを事前に止められる。",
        "",
        "## Result",
        "",
        f"- observed candidate rows: `{result['observed_rows']}`",
        f"- validation-pass but no-gain rows: `{result['validation_pass_no_gain_rows']}`",
        f"- known improved rows: `{result['improved_rows']}`",
        f"- base local estimate remains: `{result['base_local_estimate']:.6f}`",
        "",
        "## Main Guardrails",
        "",
        "- 長い `Where` chain、全画素 `Tile`、動的 `MatMul` color map、大きい座標initializer付き `ScatterND` はONNX生成前に概算cost gateへ通す。",
        "- `Slice`, `Gather`, small `Conv`, initializer pruning は優先して試す。",
        "- `signature_scatternd_lookup` はlocal upper boundとしては使えるが、7600/PB狙いではrule miningへ戻す。",
        "",
        "## Worker Audit",
        "",
        "低reasoning workerの監査でも、exp019-025の失敗は同じ傾向だった。"
        "正しいruleでもfull-grid/dynamic/large-initializer loweringはbaseline artifactに負けるため、"
        "main orchestratorがworker成果物をreviewし、cost forecast未達なら再帰的に差し戻す。",
        "",
        "## Next PDCA",
        "",
        "1. `exp028_lookup_to_rule_miner` でsignature lookupの圧縮候補を作る。",
        "2. `exp029_crop_object_synthesizer` でconstant Slice/Gather系を増やす。",
        "3. loweringごとにexpected_cost_upper_boundを持たせ、上限超過ならONNX emissionしない。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_result = load_json(BASE_RESULT_PATH)
    base_local = float(base_result.get("new_local_estimate") or base_result.get("local_estimate"))

    observations = read_observations()
    pattern_summary = summarize_patterns(observations)
    write_csv(EXP_DIR / "observed_lowerings.csv", observations)
    write_csv(EXP_DIR / "pattern_summary.csv", pattern_summary)
    write_csv(EXP_DIR / "lowering_guardrails.csv", guardrail_rows())

    validation_pass_no_gain = [
        row
        for row in observations
        if row["status"] == "no_cost_gain" and str(row["validation_status"]).endswith("_0_fail")
    ]
    improved_rows = [row for row in observations if row["status"] == "improved"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "cost_guardrails_ready",
        "base_exp": "experiments/exp023_graph_surgery_exp016",
        "base_local_estimate": base_local,
        "gap_to_6500": 6500.0 - base_local,
        "gap_to_7600": 7600.0 - base_local,
        "observed_rows": len(observations),
        "pattern_count": len(pattern_summary),
        "validation_pass_no_gain_rows": len(validation_pass_no_gain),
        "improved_rows": len(improved_rows),
        "top_validation_pass_no_gain": validation_pass_no_gain[:20],
        "pattern_summary": pattern_summary,
        "outputs": {
            "observed_lowerings": "observed_lowerings.csv",
            "pattern_summary": "pattern_summary.csv",
            "lowering_guardrails": "lowering_guardrails.csv",
        },
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic guardrails only.",
        "leakage_risk": "low for this diagnostic; high for any future memorized lookup candidate.",
        "overfitting_risk": "low for this diagnostic; future synthesis still needs full validation.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import date
from typing import Any

import onnx
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import sha256  # noqa: E402


EXP_ID = "exp100_low_cost_artifact_rule_mining"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
PROFILE_CSV = ROOT / "experiments" / "exp_b033_low_cost_artifact_profile" / "artifact_profile.csv"
CAMPAIGN_PROGRESS = ROOT / "experiments" / "compiler_campaign_30" / "progress.md"
CURRENT_LOCAL_ESTIMATE = 6282.812218


def load_profile_rows() -> list[dict[str, str]]:
    with PROFILE_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_raws() -> dict[int, bytes]:
    with zipfile.ZipFile(CURRENT_EXP / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def tensor_shape(value: onnx.ValueInfoProto) -> str:
    if not value.type.HasField("tensor_type"):
        return ""
    dims: list[str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return "x".join(dims)


def initializer_summary(model: onnx.ModelProto) -> tuple[int, int, str]:
    rows = []
    params = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        params += int(arr.size)
        rows.append(f"{init.name}:{'x'.join(map(str, arr.shape)) or 'scalar'}:{arr.dtype}:{arr.size}")
    return len(rows), params, ";".join(rows[:20])


def classify_template(op_seq: list[str], cost: int, init_params: int) -> tuple[str, str, str]:
    ops = Counter(op_seq)
    seq = " ".join(op_seq)
    if len(op_seq) == 1 and op_seq[0] == "Transpose":
        return "one_node_transpose", "free_geom", "Prefer one-node Transpose only when padded semantics match exactly."
    if len(op_seq) == 1 and op_seq[0] == "Gather":
        if init_params <= 10:
            return "channel_gather_colormap", "free_colormap", "Use Gather(axis=1) for injective one-hot channel recolor."
        return "one_node_gather_index_map", "static_index_map", "Use only if index initializer is tiny and semantics are full-padded safe."
    if len(op_seq) == 1 and op_seq[0] == "Conv":
        return "one_node_conv_kernel", "local_mask_or_colormap", "Mine weights; one Conv can encode local mask/color-role patterns under 250-1000."
    if set(op_seq) == {"Slice", "Pad"} and op_seq == ["Slice", "Pad"]:
        return "static_slice_pad", "fixed_crop_or_fixed_transform", "Best static crop archetype; cost scales with active output area."
    if "ArgMax" in ops and "ReduceSum" in ops and "Gather" in ops and "Pad" in ops and cost <= 1000:
        return "tiny_dynamic_shape_index", "shape_index_crop", "Use ReduceSum/ArgMax/Gather to select small shape/offset without full-grid masks."
    if "Conv" in ops and "Where" in ops and cost <= 1000:
        return "small_conv_where_mask", "local_mask", "Small Conv + scalar reductions can be viable; avoid full-grid repeated Where."
    if "Slice" in ops and "Pad" in ops and "Sub" in ops and cost <= 2000:
        return "computed_slice_pad", "object_anchor_crop", "Small arithmetic around Slice/Pad is viable for object-anchor crop."
    if "GridSample" in ops:
        return "one_node_gridsample", "resize_transform", "GridSample is compact but verify official allow/static behavior before reuse."
    if cost <= 600:
        return "misc_under_600", "unknown_low_cost", "Inspect manually; under-600 pattern may be reusable."
    return "misc_under_2000", "unknown_near_low_cost", "Potentially useful but not first compiler archetype."


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_profile_rows()
    raws = load_raws()
    selected = [row for row in rows if int(float(row["cost"])) <= 2000]
    catalog_rows: list[dict[str, Any]] = []
    archetype_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    rewrite_candidates: dict[str, dict[str, Any]] = {}

    for row in selected:
        task_id = int(row["task_id"])
        raw = raws[task_id]
        model = onnx.load_model_from_string(raw)
        try:
            inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        except Exception:
            inferred = model
        op_seq = [node.op_type for node in model.graph.node]
        init_count, init_params, init_detail = initializer_summary(model)
        cost = int(float(row["cost"]))
        archetype, family, lesson = classify_template(op_seq, cost, init_params)
        archetype_counts[archetype] += 1
        family_counts[family] += 1
        value_shapes = {v.name: tensor_shape(v) for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
        max_intermediate_rank = 0
        for shape in value_shapes.values():
            if shape:
                max_intermediate_rank = max(max_intermediate_rank, len(shape.split("x")))
        out_shapes = ";".join(tensor_shape(v) for v in inferred.graph.output)
        catalog_rows.append(
            {
                "task_id": task_id,
                "cost": cost,
                "route": row["route"],
                "source": row["source"],
                "node_count": row["node_count"],
                "initializer_count": init_count,
                "initializer_params": init_params,
                "op_sequence": " ".join(op_seq),
                "op_counts": json.dumps(dict(Counter(op_seq)), sort_keys=True),
                "output_shapes": out_shapes,
                "value_shape_count": len(value_shapes),
                "max_intermediate_rank": max_intermediate_rank,
                "archetype": archetype,
                "compiler_family": family,
                "compiler_lesson": lesson,
                "initializer_detail": init_detail,
                "sha256": sha256(raw),
            }
        )
        if archetype not in rewrite_candidates:
            rewrite_candidates[archetype] = {
                "archetype": archetype,
                "compiler_family": family,
                "example_task_id": task_id,
                "example_cost": cost,
                "op_sequence": op_seq,
                "lesson": lesson,
                "rewrite_direction": "",
            }

    rewrite_candidates["channel_gather_colormap"]["rewrite_direction"] = "Replace injective 1x1 one-hot color Conv or color-map table with Gather(axis=1)."
    rewrite_candidates["static_slice_pad"]["rewrite_direction"] = "For fixed crop/flip/rot on small active shape, emit one Slice with negative steps where possible plus Pad; avoid extra Gather."
    rewrite_candidates["one_node_transpose"]["rewrite_direction"] = "Use one Transpose only for full padded-safe transpose tasks; otherwise crop first but cost may exceed current artifact."
    rewrite_candidates["one_node_conv_kernel"]["rewrite_direction"] = "Try single Conv for local color-role masks before multi-node equality/Where chains."

    with (EXP_DIR / "template_catalog.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "cost",
            "route",
            "source",
            "node_count",
            "initializer_count",
            "initializer_params",
            "op_sequence",
            "op_counts",
            "output_shapes",
            "value_shape_count",
            "max_intermediate_rank",
            "archetype",
            "compiler_family",
            "compiler_lesson",
            "initializer_detail",
            "sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(catalog_rows)

    rule_rows = sorted(rewrite_candidates.values(), key=lambda item: (item["compiler_family"], item["archetype"]))
    (EXP_DIR / "rewrite_rule_candidates.json").write_text(json.dumps(rule_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "compiler_catalog_ready",
        "campaign_index": 2,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": 0.0,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "selected_cost_threshold": 2000,
        "selected_count": len(selected),
        "archetype_counts": dict(archetype_counts),
        "compiler_family_counts": dict(family_counts),
        "top_archetypes": archetype_counts.most_common(10),
        "rewrite_rule_candidate_count": len(rule_rows),
        "decision": "Use these archetypes as the first NeuroGolf-specific compiler grammar. Next experiment should instantiate static_slice_pad/channel_gather/one_node_conv candidates on unsolved small-output tasks rather than expanding generic rules.",
        "submission_decision": "no_submit: catalog only",
        "leakage_risk": "low: mines current submitted-safe artifacts, not labels.",
        "overfitting_risk": "low for catalog; future generated rules require full validation and LB calibration.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    notes = f"""# {EXP_ID}

## 目的

既存のcost<=2000 artifactをNeuroGolf専用compilerのgrammarへ変換する。単なるprofileではなく、archetypeとrewrite directionを機械可読にする。

## 結果

- selected artifacts: `{len(selected)}`
- rewrite rule candidates: `{len(rule_rows)}`
- local delta: `0.000000`
- new local estimate: `{CURRENT_LOCAL_ESTIMATE:.6f}`

## Top Archetypes

{json.dumps(archetype_counts.most_common(10), ensure_ascii=False, indent=2)}

## Decision

次は `static_slice_pad`, `channel_gather_colormap`, `one_node_conv_kernel`, `tiny_dynamic_shape_index` を候補生成器として実装し、exp087の小出力候補へ流す。

## Risk

- leakage risk: low。
- overfitting risk: catalog段階ではlow。候補生成後はfull validation必須。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    if CAMPAIGN_PROGRESS.exists():
        with CAMPAIGN_PROGRESS.open("a", encoding="utf-8") as f:
            f.write("\n| 2 | exp100_low_cost_artifact_rule_mining | 低cost artifactをcompiler archetype/rewrite候補へ変換 | 0.000000 | 6282.812218 | static_slice_pad/channel_gather/one_node_convを次の生成器にする |\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

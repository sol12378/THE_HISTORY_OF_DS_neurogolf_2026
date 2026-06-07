from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
import zipfile
from collections import Counter
from datetime import date
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import sha256  # noqa: E402


EXP_ID = "exp102_one_node_conv_kernel_mining"
EXP_DIR = ROOT / "experiments" / EXP_ID
CURRENT_EXP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union"
CATALOG_CSV = ROOT / "experiments" / "exp100_low_cost_artifact_rule_mining" / "template_catalog.csv"
CAMPAIGN_PROGRESS = ROOT / "experiments" / "compiler_campaign_30" / "progress.md"
CURRENT_LOCAL_ESTIMATE = 6282.812218


def load_catalog_rows() -> list[dict[str, str]]:
    with CATALOG_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_raws() -> dict[int, bytes]:
    with zipfile.ZipFile(CURRENT_EXP / "submission.zip") as zf:
        return {int(pathlib.Path(name).stem.replace("task", "")): zf.read(name) for name in zf.namelist()}


def summarize_conv(model: onnx.ModelProto) -> dict[str, Any] | None:
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv"]
    if len(conv_nodes) != 1:
        return None
    conv = conv_nodes[0]
    weight_name = conv.input[1] if len(conv.input) >= 2 else ""
    init_by_name = {init.name: init for init in model.graph.initializer}
    if weight_name not in init_by_name:
        return None
    weight = numpy_helper.to_array(init_by_name[weight_name])
    bias = None
    if len(conv.input) >= 3 and conv.input[2] in init_by_name:
        bias = numpy_helper.to_array(init_by_name[conv.input[2]])
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in conv.attribute}
    nonzero = int(np.count_nonzero(weight))
    unique_values = sorted({float(x) for x in np.unique(weight) if float(x) != 0.0})
    out_ch, in_ch = int(weight.shape[0]), int(weight.shape[1])
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    density = nonzero / float(weight.size)
    per_out_nonzero = np.count_nonzero(weight.reshape(out_ch, -1), axis=1)
    per_in_nonzero = np.count_nonzero(np.transpose(weight, (1, 0, 2, 3)).reshape(in_ch, -1), axis=1)
    return {
        "weight_shape": "x".join(map(str, weight.shape)),
        "bias_shape": "" if bias is None else "x".join(map(str, bias.shape)),
        "kernel": f"{kh}x{kw}",
        "out_channels": out_ch,
        "in_channels": in_ch,
        "nonzero": nonzero,
        "weight_size": int(weight.size),
        "density": density,
        "unique_nonzero_values": ",".join(map(lambda x: f"{x:g}", unique_values[:20])),
        "max_abs_weight": float(np.max(np.abs(weight))) if weight.size else 0.0,
        "groups": int(attrs.get("group", 1)),
        "pads": ",".join(map(str, attrs.get("pads", []))) if "pads" in attrs else "",
        "strides": ",".join(map(str, attrs.get("strides", []))) if "strides" in attrs else "",
        "dilations": ",".join(map(str, attrs.get("dilations", []))) if "dilations" in attrs else "",
        "per_out_nonzero_min": int(per_out_nonzero.min()) if len(per_out_nonzero) else 0,
        "per_out_nonzero_max": int(per_out_nonzero.max()) if len(per_out_nonzero) else 0,
        "per_in_nonzero_min": int(per_in_nonzero.min()) if len(per_in_nonzero) else 0,
        "per_in_nonzero_max": int(per_in_nonzero.max()) if len(per_in_nonzero) else 0,
    }


def classify_conv(row: dict[str, Any]) -> tuple[str, str]:
    kernel = row["kernel"]
    groups = int(row["groups"])
    out_ch = int(row["out_channels"])
    in_ch = int(row["in_channels"])
    density = float(row["density"])
    if kernel == "1x1" and groups == 1 and density <= 0.15:
        return "sparse_1x1_colormap", "Candidate replacement: Gather(axis=1) if one-hot/injective; otherwise keep Conv."
    if kernel == "1x1":
        return "dense_1x1_linear_colormap", "Use for non-injective or additive channel mixes; compare to Gather/MatMul."
    if groups == in_ch and out_ch == in_ch:
        return "depthwise_local_filter", "Useful for local neighborhood predicates; try one Conv before explicit Slice/Mul chains."
    if kernel in {"3x3", "5x5"} and density <= 0.25:
        return "sparse_local_pattern_detector", "Encode small spatial pattern detector as one Conv; threshold with minimal scalar ops."
    return "general_conv_kernel", "Inspect manually before compiler emission."


def main() -> None:
    started = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_catalog_rows()
    raws = load_raws()
    conv_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for row in rows:
        if "Conv" not in row["op_sequence"].split():
            continue
        task_id = int(row["task_id"])
        model = onnx.load_model_from_string(raws[task_id])
        summary = summarize_conv(model)
        if summary is None:
            continue
        conv_class, compiler_note = classify_conv(summary)
        class_counts[conv_class] += 1
        conv_rows.append(
            {
                "task_id": task_id,
                "cost": int(float(row["cost"])),
                "route": row["route"],
                "archetype": row["archetype"],
                "op_sequence": row["op_sequence"],
                "conv_class": conv_class,
                "compiler_note": compiler_note,
                "sha256": sha256(raws[task_id]),
                **summary,
            }
        )

    with (EXP_DIR / "conv_kernel_catalog.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "task_id",
            "cost",
            "route",
            "archetype",
            "op_sequence",
            "conv_class",
            "compiler_note",
            "weight_shape",
            "bias_shape",
            "kernel",
            "out_channels",
            "in_channels",
            "nonzero",
            "weight_size",
            "density",
            "unique_nonzero_values",
            "max_abs_weight",
            "groups",
            "pads",
            "strides",
            "dilations",
            "per_out_nonzero_min",
            "per_out_nonzero_max",
            "per_in_nonzero_min",
            "per_in_nonzero_max",
            "sha256",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(conv_rows)

    rules = [
        {
            "name": "single_conv_before_slice_mul_chain",
            "applies_to": ["task185_homogeneous_2x2", "local_mask_patterns"],
            "rule": "Prefer one Conv pattern detector plus minimal threshold over shifted Slice/Mul chains.",
            "evidence": "exp089 Conv proxy 1147 beat exp090 Mul proxies 2200/2848; exp102 confirms many low-cost artifacts rely on one Conv.",
        },
        {
            "name": "conv_to_gather_when_sparse_1x1_injective",
            "applies_to": ["color_map", "channel_recolor"],
            "rule": "Replace sparse injective 1x1 Conv with Gather(axis=1).",
            "evidence": "Existing channel Gather artifacts cost 10-30; Conv artifacts are usually 160+.",
        },
        {
            "name": "depthwise_conv_visibility_guardrail",
            "applies_to": ["ray_visibility", "connectivity"],
            "rule": "Do not emit large full-grid depthwise visibility Conv unless estimated cost beats current artifact.",
            "evidence": "exp_b011 task037 depthwise Conv cost 441976 despite correct rule.",
        },
    ]
    (EXP_DIR / "conv_rewrite_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "conv_kernel_catalog_ready",
        "campaign_index": 4,
        "base_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "local_estimate_delta": 0.0,
        "new_local_estimate": CURRENT_LOCAL_ESTIMATE,
        "conv_artifact_count": len(conv_rows),
        "conv_class_counts": dict(class_counts),
        "rules": rules,
        "decision": "Use one-Conv local detector as the next lowering primitive, but only with strict cost estimation and shape limits. Start with task185-like tiny lattice/local-mask tasks rather than connectivity/ray visibility.",
        "submission_decision": "no_submit: mining only",
        "leakage_risk": "low: mines current submitted-safe artifacts.",
        "overfitting_risk": "low for catalog; generated Conv rules still require full validation.",
        "runtime_seconds": time.time() - started,
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp100で最大archetypeだった `one_node_conv_kernel` を逆解析し、Convをcompiler primitiveとして使う条件とguardrailを作る。

## 結果

- conv artifacts: `{len(conv_rows)}`
- class counts: `{dict(class_counts)}`
- local delta: `0.000000`

## Decision

次のscore-producing候補は、connectivity/rayではなく、task185型の小lattice/local-maskに対して one Conv detector を使う方向にする。巨大visibility Convはguardrailで弾く。

## Risk

- leakage risk: low。
- overfitting risk: catalog段階ではlow。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    if CAMPAIGN_PROGRESS.exists():
        with CAMPAIGN_PROGRESS.open("a", encoding="utf-8") as f:
            f.write("\n| 4 | exp102_one_node_conv_kernel_mining | one-node Conv artifactを逆解析しConv compiler guardrailを作成 | 0.000000 | 6282.812218 | task185型小local-maskへone Conv primitiveを使う |\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

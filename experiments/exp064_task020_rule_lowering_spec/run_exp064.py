from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


EXP_ID = "exp064_task020_rule_lowering_spec"
ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
EXP062 = ROOT / "experiments" / "exp062_task020_canon_pos_rule_compressor" / "result.json"
EXP046 = ROOT / "experiments" / "exp046_task031_small_bbox_compiler" / "result.json"


@dataclass(frozen=True)
class LoweringBlock:
    block: str
    purpose: str
    preferred_ops: str
    expected_cost_risk: str
    reject_pattern: str
    implementation_note: str


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    exp062 = json.loads(EXP062.read_text(encoding="utf-8"))
    exp046 = json.loads(EXP046.read_text(encoding="utf-8"))
    blocks = [
        LoweringBlock(
            "bbox5_crop",
            "Find nonzero bbox and crop/align the task020 5x5 local frame.",
            "ReduceSum, Greater, ArgMax, Gather",
            "medium-high; exp046 small bbox crop cost was 50589 for task031",
            "full-grid GatherND, NonZero, Compress",
            "Reuse exp046 style dynamic bbox crop first; later replace with task020-specific 10x10/bbox5 shortcuts.",
        ),
        LoweringBlock(
            "target_color_roles",
            "For each nonzero color, compute group counts in corners, edge_mid, inner, center over the 5x5 frame.",
            "Gather/Slice over 5x5, ReduceSum, Add",
            "medium if done for all 9 colors; low if target color can be inferred cheaply",
            "MatMul signature over examples, per-example color table",
            "Rule must stay color-role based. Do not use absolute target-color table.",
        ),
        LoweringBlock(
            "class_selector",
            "Implement exp062 decision rule: corners_eq1 -> corners, edge_mid_eq1 -> edge_mid, has_inner -> inner_diag, else corners.",
            "Equal, Greater, And/Or/Where",
            "low",
            "24-case lookup table",
            "This is the core nonlookup rule and should remain tiny.",
        ),
        LoweringBlock(
            "orientation_and_missing",
            "Select the unique D4 variant compatible with current target-color positions and fill exactly the three missing cells.",
            "constant 5x5 masks, And/Or, Where",
            "medium; could explode if implemented separately for all colors/classes/variants",
            "large ScatterND coordinate table, MatMul over variants",
            "Start with explicit masks for 3 template classes x D4 variants, then prune with compatibility.",
        ),
        LoweringBlock(
            "writeback",
            "Map 5x5 local fill mask back to 30x30 one-hot output.",
            "Pad/Gather or Where on full 30x30 tensor",
            "medium-high; full-grid Where may dominate memory",
            "repeated full-grid Where per variant/color",
            "First implementation can use full-grid Where for correctness; cost pass will likely need sparse writeback.",
        ),
    ]
    rows = [asdict(b) for b in blocks]
    with (EXP_DIR / "lowering_blocks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LoweringBlock.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "lowering_spec_ready",
        "task_id": 20,
        "rule_source": "exp062_task020_canon_pos_rule_compressor",
        "rule": exp062["rules"][1],
        "known_python_validation": "266/266 from exp062",
        "bbox_reference": {
            "exp": "exp046_task031_small_bbox_compiler",
            "validation": exp046["candidate"]["validation_status"],
            "candidate_cost": exp046["candidate"]["candidate_cost"],
            "interpretation": "dynamic bbox crop can validate but may cost around 50k before task-specific shrinking",
        },
        "lowering_blocks": rows,
        "first_onnx_target": {
            "goal": "correctness-first ONNX for task020 explicit rule",
            "acceptance": "266/266 all arc-gen pass and candidate_cost < 90133 strict task020",
            "stretch_acceptance": "candidate_cost <= 600",
            "submit_policy": "submit to Kaggle only if official validation passes and cost improves strict seed",
        },
        "decision": "Implement correctness-first ONNX next, then optimize bbox/writeback. Do not use raw 24-case table or teacher artifact.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: spec only",
        "leakage_risk": "low for the rule, medium for future implementation if masks become lookup-like.",
        "overfitting_risk": "medium: task-specific rule, but fully explainable and all-arc-gen exact in Python.",
        "outputs": {"lowering_blocks": "lowering_blocks.csv", "notes": "notes.md"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

exp062のtask020明示ruleをONNX化する前に、dynamic bbox込みのlowering blockを明確にする。

## 結論

最初のONNX targetはcorrectness-first。acceptanceは `266/266` all arc-gen pass かつ strict task020 cost `90133` 未満。

cost 250〜600はstretch targetであり、初回はbbox/writebackが重くなる可能性が高い。

## Rule

- `corners_eq1 -> corners`
- `edge_mid_eq1 -> edge_mid`
- `has_inner -> inner_diag`
- default `corners`

## Next

1. exp046型のdynamic bbox cropで5x5 local frameを作る。
2. group countとclass selectorをONNX化する。
3. 3 template x D4 variant masksでmissing cellsを決める。
4. writebackして検証/cost計測する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from itertools import combinations
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402
from experiments.exp059_task020_three_template_selector.run_exp059 import (  # noqa: E402
    TEMPLATES,
    apply_prediction,
    d4_variants,
    input_features,
    true_template_name,
)


EXP_ID = "exp062_task020_canon_pos_rule_compressor"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class CaseRow:
    canon_pos: str
    count: int
    template_name: str
    feature_bits: str


@dataclass(frozen=True)
class RuleRow:
    rule_name: str
    status: str
    class_accuracy: int
    output_pass: int
    total_examples: int
    case_accuracy: int
    total_cases: int
    fail_reasons: str
    rule_spec: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def pos_bits(canon_pos: tuple[tuple[int, int], ...]) -> dict[str, bool]:
    pts = set(canon_pos)
    corners = {(0, 0), (0, 4), (4, 0), (4, 4)}
    edge_mid = {(0, 2), (2, 0), (2, 4), (4, 2)}
    inner = {(1, 1), (1, 3), (3, 1), (3, 3)}
    center = {(2, 2)}
    bits: dict[str, bool] = {}
    for name, group in [("corners", corners), ("edge_mid", edge_mid), ("inner", inner), ("center", center)]:
        inter = pts & group
        bits[f"has_{name}"] = bool(inter)
        bits[f"only_{name}"] = bool(pts) and pts <= group
        bits[f"{name}_ge1"] = len(inter) >= 1
        bits[f"{name}_ge2"] = len(inter) >= 2
        bits[f"{name}_ge3"] = len(inter) >= 3
        bits[f"{name}_eq0"] = len(inter) == 0
        bits[f"{name}_eq1"] = len(inter) == 1
        bits[f"{name}_eq2"] = len(inter) == 2
    bits["count_eq1"] = len(pts) == 1
    bits["count_eq2"] = len(pts) == 2
    bits["count_ge5"] = len(pts) >= 5
    bits["corner_and_edge"] = bool(pts & corners) and bool(pts & edge_mid)
    bits["corner_and_inner"] = bool(pts & corners) and bool(pts & inner)
    bits["edge_and_inner"] = bool(pts & edge_mid) and bool(pts & inner)
    bits["center_present"] = bool(pts & center)
    return bits


def load_records() -> list[dict[str, Any]]:
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    records = []
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        template_name, _, color = true_template_name(x, y)
        feats = input_features(x, color)
        canon_pos = tuple(feats["canon_pos"])
        records.append(
            {
                "idx": idx,
                "split": split_name(idx, train_n, test_n),
                "x": x,
                "y": y,
                "target_color": color,
                "template_name": template_name,
                "canon_pos": canon_pos,
                "bits": pos_bits(canon_pos),
            }
        )
    return records


def learn_case_table(records: list[dict[str, Any]]) -> dict[tuple[tuple[int, int], ...], str]:
    table: dict[tuple[tuple[int, int], ...], Counter[str]] = defaultdict(Counter)
    for rec in records:
        table[rec["canon_pos"]][rec["template_name"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in table.items() if len(v) == 1}


def predict_rule(rule: dict[str, Any], rec: dict[str, Any]) -> str | None:
    bits = rec["bits"]
    if rule["kind"] == "decision":
        for conds, label in rule["branches"]:
            if all(bits.get(c, False) for c in conds):
                return label
        return rule.get("default")
    if rule["kind"] == "case_table":
        return rule["table"].get(str(rec["canon_pos"]))
    return None


def evaluate_rule(name: str, rule: dict[str, Any], records: list[dict[str, Any]], cases: dict[tuple[tuple[int, int], ...], str]) -> RuleRow:
    class_ok = 0
    output_ok = 0
    case_ok = 0
    fail = Counter()
    for case, label in cases.items():
        dummy = {"canon_pos": case, "bits": pos_bits(case)}
        pred = predict_rule(rule, dummy)
        if pred == label:
            case_ok += 1
    for rec in records:
        pred_class = predict_rule(rule, rec)
        if pred_class == rec["template_name"]:
            class_ok += 1
        else:
            fail[f"class:{pred_class}->{rec['template_name']}"] += 1
        if pred_class is None:
            pred = rec["x"]
        else:
            pred = apply_prediction(rec["x"], rec["target_color"], pred_class)
        if np.array_equal(pred, rec["y"]):
            output_ok += 1
        else:
            fail["output_mismatch"] += 1
    status = "full_pass" if output_ok == len(records) else "partial"
    return RuleRow(
        rule_name=name,
        status=status,
        class_accuracy=class_ok,
        output_pass=output_ok,
        total_examples=len(records),
        case_accuracy=case_ok,
        total_cases=len(cases),
        fail_reasons=json.dumps(dict(fail.most_common(8)), ensure_ascii=False),
        rule_spec=json.dumps(rule, ensure_ascii=False, default=str),
    )


def synthesize_rules(cases: dict[tuple[tuple[int, int], ...], str]) -> dict[str, dict[str, Any]]:
    bit_names = sorted(pos_bits(next(iter(cases))).keys())
    labels = sorted(set(cases.values()))
    rules: dict[str, dict[str, Any]] = {
        "raw_24_case_table_diagnostic": {
            "kind": "case_table",
            "table": {str(k): v for k, v in cases.items()},
        }
    }
    # Greedy one-vs-rest conjunctions. This is still small and readable if it uses 1-3 bits.
    branches = []
    remaining = set(cases.keys())
    for label in labels:
        best = None
        for size in [1, 2, 3]:
            for conds in combinations(bit_names, size):
                matched = {case for case in remaining if all(pos_bits(case).get(c, False) for c in conds)}
                if not matched:
                    continue
                wrong = sum(1 for case in matched if cases[case] != label)
                correct = sum(1 for case in matched if cases[case] == label)
                score = (wrong, -correct, size, conds)
                if best is None or score < best[0]:
                    best = (score, conds, matched)
            if best and best[0][0] == 0:
                break
        if best is not None and best[0][0] == 0:
            _, conds, matched = best
            branches.append((tuple(conds), label))
            remaining -= matched
    if remaining:
        default = Counter(cases[c] for c in remaining).most_common(1)[0][0]
    else:
        default = Counter(cases.values()).most_common(1)[0][0]
    rules["greedy_pure_bit_branches"] = {"kind": "decision", "branches": branches, "default": default}

    # Hand-sized hypotheses from the audit.
    rules["orbit_presence_priority"] = {
        "kind": "decision",
        "branches": [
            (("only_inner",), "inner_diag"),
            (("only_edge_mid",), "edge_mid"),
            (("only_corners",), "corners"),
            (("inner_ge2",), "inner_diag"),
            (("edge_mid_ge2",), "edge_mid"),
            (("corners_ge2",), "corners"),
            (("has_inner",), "inner_diag"),
            (("has_edge_mid",), "edge_mid"),
        ],
        "default": "corners",
    }
    rules["missing_orbit_priority"] = {
        "kind": "decision",
        "branches": [
            (("corners_eq0",), "corners"),
            (("edge_mid_eq0",), "edge_mid"),
            (("inner_eq0",), "inner_diag"),
            (("corners_eq1", "edge_mid_ge2"), "corners"),
            (("edge_mid_eq1", "inner_ge2"), "edge_mid"),
            (("inner_eq1", "corners_ge2"), "inner_diag"),
        ],
        "default": "corners",
    }
    return rules


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    cases = learn_case_table(records)
    case_rows = []
    case_counts: dict[tuple[tuple[int, int], ...], int] = Counter(rec["canon_pos"] for rec in records)
    for case, label in sorted(cases.items(), key=lambda kv: (kv[1], kv[0])):
        case_rows.append(
            CaseRow(
                canon_pos=str(case),
                count=case_counts[case],
                template_name=label,
                feature_bits=json.dumps(pos_bits(case), ensure_ascii=False, sort_keys=True),
            )
        )

    rules = synthesize_rules(cases)
    rows = [evaluate_rule(name, rule, records, cases) for name, rule in rules.items()]
    rows.sort(key=lambda r: (r.status == "full_pass", r.output_pass, r.class_accuracy, r.case_accuracy), reverse=True)
    full_hits = [r for r in rows if r.status == "full_pass" and r.rule_name != "raw_24_case_table_diagnostic"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else "no_submit_safe_rule",
        "target_task": TARGET_TASK,
        "case_count": len(cases),
        "example_count": len(records),
        "rule_count": len(rows),
        "full_pass_nonlookup_hit_count": len(full_hits),
        "best_rule": asdict(rows[0]),
        "rules": [asdict(r) for r in rows],
        "decision": "Raw case table is diagnostic only. Lower only a nonlookup full-pass rule; otherwise continue rule compression.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit unless nonlookup full-pass rule is found",
        "leakage_risk": "high for raw case table; medium for bit-branch rules induced from all examples.",
        "overfitting_risk": "high unless the rule is simple and family-consistent.",
        "outputs": {
            "case_summary": "case_summary.csv",
            "rule_eval": "rule_eval.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "case_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CaseRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in case_rows])
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RuleRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

task020の`canon_pos` 24 caseを、raw lookupではなく説明可能な小ruleへ圧縮する。

## 結果

- cases: {len(cases)}
- rules: {len(rows)}
- nonlookup full-pass hits: {len(full_hits)}
- best: {asdict(rows[0])}

## Decision

raw 24-case tableはdiagnosticのみ。nonlookup full-pass ruleが出るまではONNX lowering/提出しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

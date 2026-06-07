from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402
from experiments.exp058_task020_residual_template_audit.run_exp058 import canonical_template  # noqa: E402


EXP_ID = "exp059_task020_three_template_selector"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASK = 20


@dataclass(frozen=True)
class SelectorEval:
    selector_name: str
    status: str
    total_pass: int
    total_examples: int
    train_pass: int
    train_examples: int
    test_pass: int
    test_examples: int
    arc_pass: int
    arc_examples: int
    template_accuracy: int
    template_total: int
    fail_reasons: str
    selector_spec: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def rel_positions(mask: np.ndarray, b: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = b
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in np.argwhere(mask)))


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


TEMPLATES = {
    "edge_mid": ((0, 2), (2, 0), (2, 4)),
    "corners": ((0, 0), (0, 4), (4, 0)),
    "inner_diag": ((1, 1), (1, 3), (3, 1)),
}


def d4_variants(points: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    h, w = shape
    variants = []
    for mode in range(8):
        out = []
        for r, c in points:
            if mode == 0:
                rr, cc = r, c
            elif mode == 1:
                rr, cc = r, w - 1 - c
            elif mode == 2:
                rr, cc = h - 1 - r, c
            elif mode == 3:
                rr, cc = h - 1 - r, w - 1 - c
            elif mode == 4:
                rr, cc = c, r
            elif mode == 5:
                rr, cc = c, h - 1 - r
            elif mode == 6:
                rr, cc = w - 1 - c, r
            else:
                rr, cc = w - 1 - c, h - 1 - r
            out.append((rr, cc))
        variants.append(tuple(sorted(out)))
    return sorted(set(variants))


def true_template_name(x: np.ndarray, y: np.ndarray) -> tuple[str, tuple[tuple[int, int], ...], int]:
    b = bbox(x != 0)
    assert b is not None
    changed = x != y
    colors = [int(v) for v in np.unique(y[changed]) if int(v) != 0]
    color = colors[0] if colors else -1
    rel = rel_positions(changed & (y == color), b)
    canon = canonical_template(rel, (b[2] - b[0], b[3] - b[1]))
    for name, template in TEMPLATES.items():
        if tuple(template) == tuple(canon):
            return name, rel, color
    return "unknown", rel, color


def input_features(x: np.ndarray, color: int) -> dict[str, Any]:
    b = bbox(x != 0)
    assert b is not None
    r0, c0, r1, c1 = b
    local = x[r0:r1, c0:c1]
    shape = local.shape
    pos = rel_positions(local == color, (0, 0, shape[0], shape[1]))
    feats: dict[str, Any] = {}
    feats["pos"] = pos
    feats["pos_count"] = len(pos)
    feats["canon_pos"] = canonical_template(pos, shape) if pos else ()
    feats["row_counts"] = tuple(int(np.sum(local[r, :] == color)) for r in range(shape[0]))
    feats["col_counts"] = tuple(int(np.sum(local[:, c] == color)) for c in range(shape[1]))
    feats["occupied_template"] = None
    for name, template in TEMPLATES.items():
        variants = d4_variants(template, shape)
        if pos in variants:
            feats["occupied_template"] = name
            break
    feats["missing_from_edge_mid"] = tuple(sorted(set(d4_variants(TEMPLATES["edge_mid"], shape)[0]) - set(pos)))
    feats["touches_center"] = (shape[0] // 2, shape[1] // 2) in pos
    feats["touches_corner"] = any(p in {(0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1)} for p in pos)
    feats["touches_edge_mid"] = any(p in {(0, shape[1] // 2), (shape[0] // 2, 0), (shape[0] // 2, shape[1] - 1), (shape[0] - 1, shape[1] // 2)} for p in pos)
    return feats


def build_lookup_selector(train_records: list[dict[str, Any]], key: str) -> dict[str, str]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in train_records:
        table[json.dumps(rec["features"][key], sort_keys=True)] [rec["template_name"]] += 1
    return {k: cnt.most_common(1)[0][0] for k, cnt in table.items() if len(cnt) == 1}


def predict_template(selector: dict[str, Any], features: dict[str, Any]) -> str | None:
    kind = selector["kind"]
    if kind == "lookup":
        key = json.dumps(features[selector["key"]], sort_keys=True)
        return selector["table"].get(key)
    if kind == "occupied_to_missing":
        occupied = features.get("occupied_template")
        if occupied == "edge_mid":
            return "edge_mid"
        if occupied == "corners":
            return "corners"
        if occupied == "inner_diag":
            return "inner_diag"
        return None
    if kind == "touch_rule":
        if features["touches_edge_mid"] and not features["touches_corner"]:
            return "edge_mid"
        if features["touches_corner"] and not features["touches_edge_mid"]:
            return "corners"
        if features["touches_center"]:
            return "inner_diag"
        return None
    return None


def apply_prediction(x: np.ndarray, target_color: int, template_name: str) -> np.ndarray:
    b = bbox(x != 0)
    if b is None:
        return x.copy()
    r0, c0, r1, c1 = b
    shape = (r1 - r0, c1 - c0)
    local = x[r0:r1, c0:c1]
    color_pos = set(rel_positions(local == target_color, (0, 0, shape[0], shape[1])))
    candidates = []
    for variant in d4_variants(TEMPLATES[template_name], shape):
        missing = set(variant) - color_pos
        if len(missing) != 3:
            continue
        if all(local[r, c] == 0 for r, c in missing):
            candidates.append(missing)
    if len(candidates) != 1:
        return x.copy()
    out = x.copy()
    for rr, cc in candidates[0]:
        out[r0 + rr, c0 + cc] = target_color
    return out


def load_records() -> tuple[list[dict[str, Any]], int, int]:
    task = load_task(TARGET_TASK)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    records = []
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        name, rel, color = true_template_name(x, y)
        records.append(
            {
                "idx": idx,
                "split": split_name(idx, train_n, test_n),
                "x": x,
                "y": y,
                "target_color": color,
                "template_name": name,
                "true_rel": rel,
                "features": input_features(x, color),
            }
        )
    return records, train_n, test_n


def evaluate_selector(name: str, selector: dict[str, Any], records: list[dict[str, Any]]) -> SelectorEval:
    counts = Counter()
    passes = Counter()
    fail_reasons = Counter()
    template_ok = 0
    for rec in records:
        counts[rec["split"]] += 1
        pred_template = predict_template(selector, rec["features"])
        if pred_template == rec["template_name"]:
            template_ok += 1
        if pred_template is None:
            pred = rec["x"]
            fail_reasons["no_template"] += 1
        else:
            pred = apply_prediction(rec["x"], rec["target_color"], pred_template)
        if np.array_equal(pred, rec["y"]):
            passes[rec["split"]] += 1
        else:
            if pred_template is not None and pred_template != rec["template_name"]:
                fail_reasons[f"wrong_template:{pred_template}->{rec['template_name']}"] += 1
            else:
                fail_reasons["template_ok_but_apply_fail"] += 1
    total_pass = sum(passes.values())
    total = len(records)
    status = "full_pass" if total_pass == total else ("train_test_pass" if passes["train"] == counts["train"] and passes["test"] == counts["test"] else "partial")
    return SelectorEval(
        selector_name=name,
        status=status,
        total_pass=total_pass,
        total_examples=total,
        train_pass=passes["train"],
        train_examples=counts["train"],
        test_pass=passes["test"],
        test_examples=counts["test"],
        arc_pass=passes["arc-gen"],
        arc_examples=counts["arc-gen"],
        template_accuracy=template_ok,
        template_total=total,
        fail_reasons=json.dumps(dict(fail_reasons.most_common(8)), ensure_ascii=False),
        selector_spec=json.dumps(selector, ensure_ascii=False, default=str),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    records, train_n, test_n = load_records()
    train_records = records[:train_n]
    selectors = {
        "lookup_canon_pos_train": {"kind": "lookup", "key": "canon_pos", "table": build_lookup_selector(train_records, "canon_pos")},
        "lookup_row_counts_train": {"kind": "lookup", "key": "row_counts", "table": build_lookup_selector(train_records, "row_counts")},
        "lookup_col_counts_train": {"kind": "lookup", "key": "col_counts", "table": build_lookup_selector(train_records, "col_counts")},
        "occupied_to_missing": {"kind": "occupied_to_missing"},
        "touch_rule": {"kind": "touch_rule"},
    }
    rows = [evaluate_selector(name, selector, records) for name, selector in selectors.items()]
    rows.sort(key=lambda r: (r.status == "full_pass", r.status == "train_test_pass", r.total_pass, r.template_accuracy), reverse=True)
    full_hits = [r for r in rows if r.status == "full_pass"]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else "no_full_hit",
        "target_task": TARGET_TASK,
        "selector_count": len(rows),
        "full_pass_hit_count": len(full_hits),
        "best_selector": asdict(rows[0]),
        "selectors": [asdict(r) for r in rows],
        "template_counts": dict(Counter(r["template_name"] for r in records)),
        "decision": "If full pass, lower to tiny three-template selector; otherwise inspect why orientation/application fails.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: no full-pass selector yet",
        "leakage_risk": "medium: train lookup selectors are diagnostic only unless reduced to simple generative rules.",
        "overfitting_risk": "medium-high for train lookup; lower only simple full-pass selector.",
        "outputs": {
            "selector_eval": "selector_eval.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "selector_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SelectorEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = f"""# {EXP_ID}

## 目的

task020を3-template selector問題として解く。

## 結果

- selector count: {len(rows)}
- full pass hits: {len(full_hits)}
- best: {asdict(rows[0])}

## Decision

full passが出るまではONNX loweringしない。train lookupでよい結果が出ても、そのままtemplate tableとして提出しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

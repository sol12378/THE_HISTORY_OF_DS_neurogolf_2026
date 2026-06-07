from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from typing import Callable

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp055_l1_sparse_fill_rule_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TAXONOMY = ROOT / "experiments" / "exp054_signature_lookup_family_taxonomy" / "task_taxonomy.csv"


@dataclass(frozen=True)
class EvalRow:
    task_id: int
    rule_name: str
    status: str
    total_pass: int
    total_examples: int
    train_pass: int
    train_examples: int
    test_pass: int
    test_examples: int
    arc_pass: int
    arc_examples: int
    fail_reasons: str
    estimated_lowering: str
    target_cost_band: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def count_window(mask: np.ndarray, radius: int, include_center: bool = False) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.int64)
    padded = np.pad(mask.astype(np.int64), radius)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if not include_center and dr == 0 and dc == 0:
                continue
            out += padded[radius + dr : radius + dr + h, radius + dc : radius + dc + w]
    return out


def count_cross(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=np.int64)
    padded = np.pad(mask.astype(np.int64), 1)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        out += padded[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return out


def diag_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    d1_has = np.zeros((h, w), dtype=bool)
    d2_has = np.zeros((h, w), dtype=bool)
    for key in np.unique(rows - cols):
        d1_has |= ((rows - cols) == key) & bool(np.any(mask[(rows - cols) == key]))
    for key in np.unique(rows + cols):
        d2_has |= ((rows + cols) == key) & bool(np.any(mask[(rows + cols) == key]))
    return d1_has, d2_has


def d4_orbit(pos: tuple[int, int], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    r, c = pos
    dr, dc = r - cr, c - cc
    pts_float = {
        (cr + dr, cc + dc),
        (cr + dr, cc - dc),
        (cr - dr, cc + dc),
        (cr - dr, cc - dc),
        (cr + dc, cc + dr),
        (cr + dc, cc - dr),
        (cr - dc, cc + dr),
        (cr - dc, cc - dr),
    }
    pts = set()
    for rr, cc2 in pts_float:
        if abs(rr - round(rr)) < 1e-9 and abs(cc2 - round(cc2)) < 1e-9:
            ri, ci = int(round(rr)), int(round(cc2))
            if 0 <= ri < h and 0 <= ci < w:
                pts.add((ri, ci))
    return frozenset(pts)


def propose_by_color(x: np.ndarray, pred_masks: Callable[[np.ndarray, int], np.ndarray]) -> np.ndarray:
    proposals: list[tuple[int, np.ndarray]] = []
    for color in [int(v) for v in np.unique(x) if int(v) != 0]:
        mask = pred_masks(x, color) & (x == 0)
        if np.any(mask):
            proposals.append((color, mask))
    out = x.copy()
    claim_count = np.zeros(x.shape, dtype=np.int64)
    claim_color = np.zeros(x.shape, dtype=np.int64)
    for color, mask in proposals:
        claim_count[mask] += 1
        claim_color[mask] = color
    out[claim_count == 1] = claim_color[claim_count == 1]
    return out


def rule_row_col_intersection(x: np.ndarray, color: int) -> np.ndarray:
    cm = x == color
    return np.repeat(cm.any(axis=1)[:, None], x.shape[1], axis=1) & np.repeat(cm.any(axis=0)[None, :], x.shape[0], axis=0)


def rule_row_or_col_two_sided(x: np.ndarray, color: int) -> np.ndarray:
    cm = x == color
    left = np.maximum.accumulate(cm, axis=1)
    right = np.maximum.accumulate(cm[:, ::-1], axis=1)[:, ::-1]
    up = np.maximum.accumulate(cm, axis=0)
    down = np.maximum.accumulate(cm[::-1, :], axis=0)[::-1, :]
    return (left & right) | (up & down)


def rule_color_bbox(x: np.ndarray, color: int) -> np.ndarray:
    b = bbox(x == color)
    out = np.zeros(x.shape, dtype=bool)
    if b is None:
        return out
    r0, c0, r1, c1 = b
    out[r0:r1, c0:c1] = True
    return out


def rule_nonzero_bbox_boundary(x: np.ndarray, color: int) -> np.ndarray:
    b = bbox(x != 0)
    out = np.zeros(x.shape, dtype=bool)
    if b is None:
        return out
    r0, c0, r1, c1 = b
    out[r0:r1, c0] = True
    out[r0:r1, c1 - 1] = True
    out[r0, c0:c1] = True
    out[r1 - 1, c0:c1] = True
    return out & (x == 0)


def make_neighbor_rule(kind: str, threshold: int) -> Callable[[np.ndarray, int], np.ndarray]:
    def _rule(x: np.ndarray, color: int) -> np.ndarray:
        cm = x == color
        if kind == "n4":
            return count_cross(cm) >= threshold
        if kind == "n8":
            return count_window(cm, 1) >= threshold
        if kind == "n24":
            return count_window(cm, 2) >= threshold
        raise ValueError(kind)

    return _rule


def rule_diag_intersection(x: np.ndarray, color: int) -> np.ndarray:
    d1, d2 = diag_masks(x == color)
    return d1 & d2


def rule_d4_orbit_complete(x: np.ndarray, color: int) -> np.ndarray:
    b = bbox(x != 0)
    out = np.zeros(x.shape, dtype=bool)
    if b is None:
        return out
    r0, c0, r1, c1 = b
    h, w = r1 - r0, c1 - c0
    if h != w or h > 12:
        return out
    local = x[r0:r1, c0:c1]
    pos = {(int(r), int(c)) for r, c in np.argwhere(local == color)}
    if not pos:
        return out
    for orbit in {d4_orbit((r, c), (h, w)) for r in range(h) for c in range(w)}:
        if len(orbit) <= len(pos):
            continue
        if pos <= orbit:
            missing = orbit - pos
            for rr, cc in missing:
                if local[rr, cc] != 0:
                    return np.zeros(x.shape, dtype=bool)
            if 1 <= len(missing) <= 12:
                for rr, cc in missing:
                    out[r0 + rr, c0 + cc] = True
                return out
    return out


def candidate_rules() -> dict[str, Callable[[np.ndarray, int], np.ndarray]]:
    rules: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "row_col_intersection": rule_row_col_intersection,
        "row_or_col_two_sided": rule_row_or_col_two_sided,
        "color_bbox": rule_color_bbox,
        "nonzero_bbox_boundary": rule_nonzero_bbox_boundary,
        "diag_intersection": rule_diag_intersection,
        "d4_orbit_complete": rule_d4_orbit_complete,
    }
    for kind, max_k in [("n4", 4), ("n8", 8), ("n24", 12)]:
        for threshold in range(1, max_k + 1):
            rules[f"{kind}_ge_{threshold}"] = make_neighbor_rule(kind, threshold)
    return rules


def load_l1_tasks() -> list[int]:
    with TAXONOMY.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["task_id"]) for r in rows if r["compiler_lane"] == "L1_static_sparse_background_fill"]


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def evaluate_task_rule(task_id: int, rule_name: str, rule: Callable[[np.ndarray, int], np.ndarray]) -> EvalRow:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    counts = Counter()
    passes = Counter()
    fail_reasons = Counter()
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = propose_by_color(x, rule)
        counts[split] += 1
        if np.array_equal(pred, y):
            passes[split] += 1
        else:
            if pred.shape != y.shape:
                fail_reasons["shape"] += 1
            else:
                missing = int(np.logical_and(pred == x, y != x).sum())
                extra = int(np.logical_and(pred != x, y == x).sum())
                wrong = int(np.logical_and(pred != y, pred != x, y != x).sum())
                fail_reasons[f"missing={missing};extra={extra};wrong={wrong}"] += 1
    total_pass = sum(passes.values())
    total_examples = len(examples)
    status = "full_pass" if total_pass == total_examples else ("train_test_pass" if passes["train"] == counts["train"] and passes["test"] == counts["test"] else "partial")
    return EvalRow(
        task_id=task_id,
        rule_name=rule_name,
        status=status,
        total_pass=total_pass,
        total_examples=total_examples,
        train_pass=passes["train"],
        train_examples=counts["train"],
        test_pass=passes["test"],
        test_examples=counts["test"],
        arc_pass=passes["arc-gen"],
        arc_examples=counts["arc-gen"],
        fail_reasons=json.dumps(dict(fail_reasons.most_common(5)), ensure_ascii=False),
        estimated_lowering="Equal color masks + small Reduce/Conv/constant masks + one Where",
        target_cost_band="250-600 if rule is full-pass and emitted without full-grid tables",
    )


def notes_text(result: dict[str, object]) -> str:
    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "L1 static sparse background fill 8 taskに対し、低cost lowering可能な明示ruleを探索する。",
        "",
        "## 結果",
        "",
        f"- scanned tasks: {result['scanned_task_count']}",
        f"- evaluated rules: {result['evaluated_rule_count']}",
        f"- full pass hits: {result['full_pass_hit_count']}",
        f"- train/test pass hits: {result['train_test_pass_hit_count']}",
        "",
        "## Best Partial",
        "",
    ]
    best = result["best_partial"]
    lines.append(f"- {best}")
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "full pass hitがあれば、次はそのruleだけを小さいONNXへloweringする。"
            "hitがなければ、L1でも単純な行列/近傍/対称性だけでは不足なので、object-role predicateを追加する。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_l1_tasks()
    rules = candidate_rules()
    rows: list[EvalRow] = []
    for task_id in tasks:
        for rule_name, rule in rules.items():
            rows.append(evaluate_task_rule(task_id, rule_name, rule))
    rows.sort(key=lambda r: (r.status == "full_pass", r.status == "train_test_pass", r.total_pass / r.total_examples, r.train_pass), reverse=True)
    full_hits = [r for r in rows if r.status == "full_pass"]
    train_test_hits = [r for r in rows if r.status in {"full_pass", "train_test_pass"}]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "rule_found" if full_hits else "no_full_hit",
        "source": str(TAXONOMY.relative_to(ROOT)),
        "scanned_task_count": len(tasks),
        "tasks": tasks,
        "candidate_rule_count": len(rules),
        "evaluated_rule_count": len(rows),
        "full_pass_hit_count": len(full_hits),
        "train_test_pass_hit_count": len(train_test_hits),
        "full_pass_hits": [asdict(r) for r in full_hits[:20]],
        "train_test_hits": [asdict(r) for r in train_test_hits[:20]],
        "best_partial": asdict(rows[0]) if rows else {},
        "decision": "Lower full-pass hits first; if none, add object-role predicates before ONNX emission.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: symbolic rule mining only",
        "leakage_risk": "low: generic geometric/local rules only, no teacher label table.",
        "overfitting_risk": "medium: all arc-gen is used for diagnostics; accepted ONNX candidates need holdout/calibration.",
        "outputs": {
            "rule_eval": "rule_eval.csv",
            "notes": "notes.md",
        },
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "rule_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    (EXP_DIR / "notes.md").write_text(notes_text(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import pathlib
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
EXP_ID = "exp031_zero_component_fill_miner"
EXP_DIR = ROOT / "experiments" / EXP_ID
TARGET_TASKS = [198, 187, 203, 313]
ARC_GEN_SAMPLE = 20


@dataclass
class RuleResult:
    task_id: int
    status: str
    border_color: int | str
    enclosed_color: int | str
    train_pass: int
    train_fail: int
    test_pass: int
    test_fail: int
    arc_pass: int
    arc_fail: int
    reason: str


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def zero_components(x: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = x.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[list[tuple[int, int]], bool]] = []
    for sr in range(h):
        for sc in range(w):
            if x[sr, sc] != 0 or seen[sr, sc]:
                continue
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                touches_border = touches_border or r == 0 or c == 0 or r == h - 1 or c == w - 1
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and x[nr, nc] == 0:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append((cells, touches_border))
    return out


def infer_colors(examples: list[dict[str, Any]]) -> tuple[int | None, int | None, str]:
    border_votes: Counter[int] = Counter()
    enclosed_votes: Counter[int] = Counter()
    for ex in examples:
        x = np.asarray(ex["input"], dtype=np.int64)
        y = np.asarray(ex["output"], dtype=np.int64)
        if x.shape != y.shape:
            return None, None, "shape mismatch"
        for cells, touches_border in zero_components(x):
            vals = [int(y[r, c]) for r, c in cells]
            counts = Counter(vals)
            if len(counts) != 1:
                return None, None, f"component has mixed output colors {counts}"
            color = vals[0]
            if touches_border:
                border_votes[color] += len(cells)
            else:
                enclosed_votes[color] += len(cells)
        if not np.array_equal(x[x != 0], y[x != 0]):
            return None, None, "wall cells are not preserved"
    if not border_votes:
        return None, None, "no border zero component"
    border_color = border_votes.most_common(1)[0][0]
    enclosed_color = enclosed_votes.most_common(1)[0][0] if enclosed_votes else border_color
    if len(border_votes) > 1 or len(enclosed_votes) > 1:
        return None, None, f"conflicting colors border={dict(border_votes)} enclosed={dict(enclosed_votes)}"
    return border_color, enclosed_color, "ok"


def apply_rule(x: np.ndarray, border_color: int, enclosed_color: int) -> np.ndarray:
    out = x.copy()
    for cells, touches_border in zero_components(x):
        color = border_color if touches_border else enclosed_color
        for r, c in cells:
            out[r, c] = color
    return out


def check_examples(examples: list[dict[str, Any]], border_color: int, enclosed_color: int) -> tuple[int, int, str]:
    passed = 0
    failed = 0
    reason = "ok"
    for idx, ex in enumerate(examples):
        x = np.asarray(ex["input"], dtype=np.int64)
        y = np.asarray(ex["output"], dtype=np.int64)
        pred = apply_rule(x, border_color, enclosed_color)
        if np.array_equal(pred, y):
            passed += 1
        else:
            failed += 1
            reason = f"mismatch example {idx}"
            break
    return passed, failed, reason


def evaluate_task(task_id: int) -> RuleResult:
    task = load_task(task_id)
    border_color, enclosed_color, reason = infer_colors(task["train"])
    if border_color is None or enclosed_color is None:
        return RuleResult(task_id, "no_rule", "", "", 0, len(task["train"]), 0, 0, 0, 0, reason)
    tr_p, tr_f, tr_reason = check_examples(task["train"], border_color, enclosed_color)
    te_p, te_f, te_reason = check_examples(task["test"], border_color, enclosed_color)
    arc_p, arc_f, arc_reason = check_examples(task["arc-gen"][:ARC_GEN_SAMPLE], border_color, enclosed_color)
    status = "pass_sample20" if tr_f == 0 and te_f == 0 and arc_f == 0 else "rejected"
    fail_reason = "ok"
    if tr_f:
        fail_reason = tr_reason
    elif te_f:
        fail_reason = te_reason
    elif arc_f:
        fail_reason = arc_reason
    return RuleResult(
        task_id,
        status,
        border_color,
        enclosed_color,
        tr_p,
        tr_f,
        te_p,
        te_f,
        arc_p,
        arc_f,
        fail_reason,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_task(task_id) for task_id in TARGET_TASKS]
    pass_rows = [row for row in rows if row.status == "pass_sample20"]
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "rule_found" if pass_rows else "no_gain",
        "target_tasks": TARGET_TASKS,
        "arc_gen_sample": ARC_GEN_SAMPLE,
        "pass_task_count": len(pass_rows),
        "pass_tasks": [row.task_id for row in pass_rows],
        "rows": [asdict(row) for row in rows],
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: Python rule miner only, no ONNX candidate generated.",
        "leakage_risk": "low-medium: colors are inferred from train only; validation uses test and arc-gen sample20.",
        "overfitting_risk": "medium: component-fill rule may still be ARC-family specific and sample20 is not private-like validation.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        f"# {EXP_ID}",
        "",
        "## Hypothesis",
        "",
        "入力の非ゼロセルを壁とみなし、0 connected componentのうち外周に接続する成分を外側色、閉じた成分を内側色で塗れば、region/line fill系の上位taskを説明できる。",
        "",
        "## Result",
        "",
        f"- target tasks: `{TARGET_TASKS}`",
        f"- arc-gen sample: `{ARC_GEN_SAMPLE}`",
        f"- pass tasks: `{result['pass_tasks']}`",
        "- local estimate delta: `0.0`",
        "- submission: no submit, ONNX未生成",
        "",
        "## Interpretation",
        "",
        "Python ruleとして成立したtaskだけ、次の実験でcost-aware ONNX loweringを検討する。過去のunrolled flood-fillは高costだったため、ONNX化は閉形式のrow/column prefixまたは既存artifact surgeryを優先する。",
        "",
        "## Risks",
        "",
        f"- leakage risk: {result['leakage_risk']}",
        f"- overfitting risk: {result['overfitting_risk']}",
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

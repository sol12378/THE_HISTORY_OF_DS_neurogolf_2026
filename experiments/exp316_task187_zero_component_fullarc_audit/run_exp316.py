from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
EXP_ID = "exp316_task187_zero_component_fullarc_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 187


def zero_components(x: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = x.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool]] = []
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
            components.append((cells, touches_border))
    return components


def infer_colors(train: list[dict[str, Any]]) -> tuple[int, int]:
    border_votes: Counter[int] = Counter()
    enclosed_votes: Counter[int] = Counter()
    for ex in train:
        x = np.asarray(ex["input"], dtype=np.int64)
        y = np.asarray(ex["output"], dtype=np.int64)
        for cells, touches_border in zero_components(x):
            colors = Counter(int(y[r, c]) for r, c in cells)
            if len(colors) != 1:
                raise ValueError(f"mixed output colors in one zero component: {colors}")
            color = next(iter(colors))
            if touches_border:
                border_votes[color] += len(cells)
            else:
                enclosed_votes[color] += len(cells)
        if not np.array_equal(x[x != 0], y[x != 0]):
            raise ValueError("nonzero wall cells are not preserved")
    if len(border_votes) != 1 or len(enclosed_votes) != 1:
        raise ValueError(f"non-unique colors border={dict(border_votes)} enclosed={dict(enclosed_votes)}")
    return border_votes.most_common(1)[0][0], enclosed_votes.most_common(1)[0][0]


def apply_rule(x: np.ndarray, border_color: int, enclosed_color: int) -> np.ndarray:
    y = x.copy()
    for cells, touches_border in zero_components(x):
        color = border_color if touches_border else enclosed_color
        for r, c in cells:
            y[r, c] = color
    return y


def check_split(name: str, examples: list[dict[str, Any]], border_color: int, enclosed_color: int) -> dict[str, object]:
    passed = 0
    first_failure: dict[str, object] | None = None
    for idx, ex in enumerate(examples):
        x = np.asarray(ex["input"], dtype=np.int64)
        y = np.asarray(ex["output"], dtype=np.int64)
        pred = apply_rule(x, border_color, enclosed_color)
        if np.array_equal(pred, y):
            passed += 1
            continue
        diff = np.argwhere(pred != y)
        first_failure = {
            "split": name,
            "index": idx,
            "diff_count": int(len(diff)),
            "first_diff": diff[0].tolist() if len(diff) else None,
        }
        break
    failed = 0 if first_failure is None else len(examples) - passed
    return {
        "split": name,
        "total": len(examples),
        "passed": passed,
        "failed": failed,
        "status": f"{passed}_pass_{failed}_fail",
        "first_failure": first_failure,
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = json.loads((DATA_DIR / f"task{TASK_ID:03d}.json").read_text(encoding="utf-8"))
    border_color, enclosed_color = infer_colors(task["train"])
    splits = [
        check_split("train", task["train"], border_color, enclosed_color),
        check_split("test", task["test"], border_color, enclosed_color),
        check_split("arc-gen", task["arc-gen"], border_color, enclosed_color),
    ]
    full_pass = all(row["failed"] == 0 for row in splits)
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-11",
        "status": "fullarc_rule_pass" if full_pass else "rule_failed_fullarc",
        "task_id": TASK_ID,
        "hypothesis": "task187 public-zero can be repaired by the zero-component border/enclosed fill rule from exp031",
        "inferred_rule": {"border_zero_component_color": border_color, "enclosed_zero_component_color": enclosed_color},
        "splits": splits,
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit_python_rule_audit_only; proceed to correctness-first ONNX lowering if fullarc_rule_pass",
        "leakage_risk": "low-medium: colors inferred from train only; full arc-gen is used for validation, not fitting.",
        "overfitting_risk": "medium: Python rule full-arc pass is necessary but public-zero repair still needs ONNX and LB probe.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        f"# {EXP_ID}",
        "",
        "## Plan",
        "",
        "exp031 の task187 zero-component fill rule を全 arc-gen で再検証し、correctness-first ONNX repair へ進む条件を確認する。",
        "",
        "## Result",
        "",
        f"- status: `{result['status']}`",
        f"- border zero component color: `{border_color}`",
        f"- enclosed zero component color: `{enclosed_color}`",
    ]
    for row in splits:
        notes.append(f"- {row['split']}: `{row['status']}`")
    notes.extend(
        [
            "",
            "## Act",
            "",
            str(result["submission_decision"]),
            "",
            "## Risk",
            "",
            f"- leakage risk: {result['leakage_risk']}",
            f"- overfitting risk: {result['overfitting_risk']}",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp076_task071_recolor_copy_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 71


@dataclass(frozen=True)
class FeaturePurity:
    feature: str
    keys: int
    cells: int
    majority_correct: int
    accuracy: float
    ambiguous_keys: int
    top_ambiguous: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def component_id_map(x: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    comp_id = np.full(x.shape, -1, dtype=np.int64)
    comps: list[list[tuple[int, int]]] = []
    for comp in components(x != 0):
        cid = len(comps)
        comps.append(comp)
        for r, c in comp:
            comp_id[r, c] = cid
    return comp_id, comps


def neighbor_values(x: np.ndarray, r: int, c: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, dr, dc in (("u", -1, 0), ("d", 1, 0), ("l", 0, -1), ("r", 0, 1)):
        rr, cc = r + dr, c + dc
        out[name] = int(x[rr, cc]) if 0 <= rr < x.shape[0] and 0 <= cc < x.shape[1] else -1
    return out


def color_ranks(x: np.ndarray) -> dict[int, int]:
    counts = Counter(int(v) for v in x.ravel() if int(v) != 0)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {color: idx for idx, (color, _) in enumerate(ranked)}


def collect_changed_cells(task: dict[str, Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    for idx, ex in enumerate(examples):
        split = "train" if idx < train_n else "test" if idx < train_n + test_n else "arc-gen"
        x = arr(ex["input"])
        y = arr(ex["output"])
        comp_ids, comps = component_id_map(x)
        ranks = color_ranks(x)
        for r, c in np.argwhere(x != y):
            r, c = int(r), int(c)
            cid = int(comp_ids[r, c])
            comp = comps[cid] if cid >= 0 else [(r, c)]
            r0, c0, r1, c1 = bbox(comp)
            n = neighbor_values(x, r, c)
            in_color = int(x[r, c])
            out_color = int(y[r, c])
            rows.append(
                {
                    "idx": idx,
                    "split": split,
                    "r": r,
                    "c": c,
                    "in_color": in_color,
                    "out_color": out_color,
                    "is_zero": int(out_color == 0),
                    "dr": r - r0,
                    "dc": c - c0,
                    "h": r1 - r0,
                    "w": c1 - c0,
                    "row_from_bottom": r1 - 1 - r,
                    "col_from_right": c1 - 1 - c,
                    "row_parity": (r - r0) % 2,
                    "col_parity": (c - c0) % 2,
                    "rank": ranks.get(in_color, -1),
                    "u": n["u"],
                    "d": n["d"],
                    "l": n["l"],
                    "rr": n["r"],
                    "ud_pair": f'{n["u"]}:{n["d"]}',
                    "lr_pair": f'{n["l"]}:{n["r"]}',
                    "nbr4": f'{n["u"]}:{n["d"]}:{n["l"]}:{n["r"]}',
                }
            )
    return rows


def purity(rows: list[dict[str, int | str]], fields: list[str], target: str = "out_color") -> FeaturePurity:
    buckets: dict[tuple[int | str, ...], Counter[int | str]] = defaultdict(Counter)
    for row in rows:
        buckets[tuple(row[f] for f in fields)][row[target]] += 1
    majority_correct = sum(counter.most_common(1)[0][1] for counter in buckets.values())
    ambiguous = []
    for key, counter in buckets.items():
        if len(counter) > 1:
            ambiguous.append((key, dict(counter), sum(counter.values())))
    ambiguous.sort(key=lambda x: (-x[2], str(x[0])))
    top = [
        {"key": "|".join(map(str, key)), "hist": hist, "n": n}
        for key, hist, n in ambiguous[:8]
    ]
    return FeaturePurity(
        feature="+".join(fields),
        keys=len(buckets),
        cells=len(rows),
        majority_correct=majority_correct,
        accuracy=majority_correct / len(rows) if rows else 0.0,
        ambiguous_keys=len(ambiguous),
        top_ambiguous=json.dumps(top, ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    rows = collect_changed_cells(task)
    feature_sets = [
        ["dr", "dc", "h", "w"],
        ["dr", "dc", "h", "w", "in_color"],
        ["dr", "dc", "h", "w", "rank"],
        ["row_parity", "col_parity", "in_color"],
        ["dr", "dc", "nbr4"],
        ["dr", "dc", "ud_pair", "lr_pair"],
        ["in_color", "u", "d", "l", "rr"],
        ["rank", "u", "d", "l", "rr"],
        ["dr", "dc", "h", "w", "in_color", "nbr4"],
    ]
    purities = [purity(rows, fields) for fields in feature_sets]
    zero_purities = [purity(rows, fields, target="is_zero") for fields in feature_sets]
    best = max(purities, key=lambda p: p.accuracy)
    best_zero = max(zero_purities, key=lambda p: p.accuracy)

    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "profile_ready",
        "hypothesis": "task071のmulti-color editはcomponent-local座標、入力色role、近傍色から小branch treeへ圧縮できる可能性がある。",
        "changed_cells": len(rows),
        "best_out_color_feature": asdict(best),
        "best_zero_mask_feature": asdict(best_zero),
        "decision": "出力色そのものは局所featureだけでは未解決。zero/nonzero maskは高純度ならmask分離後にcolor-copy側を別compilerへ渡す。",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: diagnostic only; no ONNX emitted",
        "next_action": "zero maskとnonzero recolorを分離し、maskだけ低cost化できるか、またはsource color copy directionを追加して再探索する。",
        "leakage_risk": "low: 全arc-genのchanged-cell feature purity診断のみで、raw key lookup候補は提出しない。",
        "overfitting_risk": "medium: 高純度featureがあってもkey数が多い場合はbranch圧縮なしでは採用しない。",
    }

    with (EXP_DIR / "changed_cell_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (EXP_DIR / "feature_purity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FeaturePurity.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in purities:
            writer.writerow(asdict(row))
    with (EXP_DIR / "zero_mask_purity.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FeaturePurity.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in zero_purities:
            writer.writerow(asdict(row))
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task071` のmulti-color recolor/copyを、低cost ONNXに落とせる小branch tree候補へ分解する。

## 仮説

変更セルの出力色またはzero maskは、component-local座標、入力色rank、4近傍色から高純度に決まる。

## 結果

- changed cells: `{len(rows)}`
- best out_color feature: `{best.feature}` accuracy `{best.accuracy:.4f}` keys `{best.keys}` ambiguous `{best.ambiguous_keys}`
- best zero-mask feature: `{best_zero.feature}` accuracy `{best_zero.accuracy:.4f}` keys `{best_zero.keys}` ambiguous `{best_zero.ambiguous_keys}`

## 判断

この実験は診断のみで、提出候補は生成していない。
出力色featureが低純度なら、`task071` は直接の小ONNX化ではなく、zero maskとsource color copy directionを分けたcompilerへ回す。

## Risk

- leakage risk: low。全arc-genを使う診断だが、raw key lookupを採用していない。
- overfitting risk: medium。高純度でもkey数が多いfeatureはbranch圧縮なしでは採用しない。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

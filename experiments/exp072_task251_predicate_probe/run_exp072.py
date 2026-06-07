from __future__ import annotations

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


EXP_ID = "exp072_task251_predicate_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 251


@dataclass(frozen=True)
class ProbeResult:
    name: str
    tp: int
    fp: int
    fn: int
    tn: int
    failed_examples: int
    first_fail: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def zero_cell_features(x: np.ndarray, r: int, c: int) -> dict[str, object]:
    h, w = x.shape
    nb = bbox(x != 0)
    adj_values: list[int] = []
    ray_values: list[int] = []
    ray_distances: list[int] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        adj_values.append(int(x[nr, nc]) if 0 <= nr < h and 0 <= nc < w else -1)
        cr, cc = r + dr, c + dc
        dist = 0
        value = 0
        while 0 <= cr < h and 0 <= cc < w:
            dist += 1
            if x[cr, cc] != 0:
                value = int(x[cr, cc])
                break
            cr += dr
            cc += dc
        ray_values.append(value)
        ray_distances.append(dist if value else 0)
    if nb is None:
        inside = False
        border = False
        dr_min = 99
        dc_min = 99
    else:
        r0, c0, r1, c1 = nb
        inside = r0 <= r < r1 and c0 <= c < c1
        border = inside and (r in {r0, r1 - 1} or c in {c0, c1 - 1})
        dr_min = min(abs(r - r0), abs(r - (r1 - 1)))
        dc_min = min(abs(c - c0), abs(c - (c1 - 1)))
    adj_nz = sum(v != 0 and v != -1 for v in adj_values)
    adj_pattern = tuple(int(v != 0 and v != -1) for v in adj_values)
    return {
        "inside": inside,
        "border": border,
        "adj_nz": adj_nz,
        "adj_distinct": len({v for v in adj_values if v != 0 and v != -1}),
        "adj_pattern": adj_pattern,
        "ray_dirs": sum(v != 0 for v in ray_values),
        "ray_values": tuple(ray_values),
        "ray_distances": tuple(ray_distances),
        "dr_min": dr_min,
        "dc_min": dc_min,
    }


def probe(name: str, pred: Callable[[dict[str, object]], bool]) -> ProbeResult:
    examples = examples_for(load_task(TASK_ID), -1)
    tp = fp = fn = tn = 0
    failed: set[int] = set()
    first_fail = ""
    for idx, ex in enumerate(examples):
        x = arr(ex["input"])
        y = arr(ex["output"])
        diff = x != y
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                if x[r, c] != 0:
                    continue
                p = bool(pred(zero_cell_features(x, r, c)))
                t = bool(diff[r, c])
                if p and t:
                    tp += 1
                elif p and not t:
                    fp += 1
                    failed.add(idx)
                    if not first_fail:
                        first_fail = json.dumps({"example": idx, "r": r, "c": c, "kind": "fp"}, ensure_ascii=False)
                elif (not p) and t:
                    fn += 1
                    failed.add(idx)
                    if not first_fail:
                        first_fail = json.dumps({"example": idx, "r": r, "c": c, "kind": "fn"}, ensure_ascii=False)
                else:
                    tn += 1
    return ProbeResult(name, tp, fp, fn, tn, len(failed), first_fail)


def fp_signature_audit() -> dict[str, object]:
    examples = examples_for(load_task(TASK_ID), -1)
    fp_sigs: Counter[str] = Counter()
    pos_sigs: Counter[str] = Counter()
    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        diff = x != y
        for r in range(x.shape[0]):
            for c in range(x.shape[1]):
                if x[r, c] != 0:
                    continue
                f = zero_cell_features(x, r, c)
                base = bool(f["inside"] and f["ray_dirs"] == 4 and int(f["adj_nz"]) >= 1)
                if not base:
                    continue
                key = str((f["adj_pattern"], f["ray_distances"], f["dr_min"], f["dc_min"]))
                if diff[r, c]:
                    pos_sigs[key] += 1
                else:
                    fp_sigs[key] += 1
    return {
        "positive_signature_count": len(pos_sigs),
        "false_positive_signature_count": len(fp_sigs),
        "positive_top": [{"key": k, "count": v} for k, v in pos_sigs.most_common(8)],
        "false_positive_top": [{"key": k, "count": v} for k, v in fp_sigs.most_common(8)],
    }


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    probes = [
        probe("inside_ray4_adj1plus", lambda f: bool(f["inside"] and f["ray_dirs"] == 4 and int(f["adj_nz"]) >= 1)),
        probe("inside_ray4_adj2", lambda f: bool(f["inside"] and f["ray_dirs"] == 4 and int(f["adj_nz"]) == 2)),
        probe("inside_ray4_adj12_dist1", lambda f: bool(f["inside"] and f["ray_dirs"] == 4 and int(f["adj_nz"]) in {1, 2} and int(f["adj_distinct"]) == 1)),
        probe(
            "reject_straight_adj2",
            lambda f: bool(
                f["inside"]
                and f["ray_dirs"] == 4
                and int(f["adj_nz"]) >= 1
                and not (
                    int(f["adj_nz"]) == 2
                    and (
                        tuple(f["adj_pattern"]) in {
                            (1, 1, 0, 0),
                            (0, 0, 1, 1),
                        }
                    )
                )
            ),
        ),
    ]
    audit = fp_signature_audit()
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "probe_ready",
        "task_id": TASK_ID,
        "hypothesis": "task251は0-cellのbbox/ray/adjacency predicateでほぼ説明でき、残りFPを小branchで削れる。",
        "probe_results": [asdict(row) for row in probes],
        "fp_signature_audit": audit,
        "decision": "base predicateは全positiveを拾うが113 FPが残る。straight adjacencyの単純rejectはpositiveを落とすため、次はray distance/local component contextを含むbranch tree synthesisが必要。",
        "leakage_risk": "low: rule predicate probe only; no lookup table or submission artifact generated.",
        "overfitting_risk": "medium: FP signatures are mostly singleton, so raw signature memorization is prohibited.",
        "outputs": {"result_json": "result.json"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp071` で最優先候補になった `task251` について、単純なchanged-cell predicateでどこまで説明できるかを測る。

## 結果

- `inside_ray4_adj1plus`: TP 2708 / FP 113 / FN 0
- `reject_straight_adj2`: FPは消える方向だがFNが出るため不採用

## 解釈

task251は `inside nonzero bbox + 4方向rayで非ゼロに挟まれる + 隣接非ゼロあり` で全positiveを拾える。残り113 false positiveだけを削る問題に縮んだ。

ただしFP signatureはsingletonが多く、raw signature table化は高risk。次はray distanceとlocal component contextを使った小branch treeを合成する。

## リスク

- leakage risk: low。提出物なし。
- overfitting risk: medium。signature memorizationを禁止し、説明可能なbranchのみ採用する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

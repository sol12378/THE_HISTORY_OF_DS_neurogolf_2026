from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp084_task365_object_crop_rule_probe"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 365


@dataclass(frozen=True)
class CandidateEval:
    candidate: str
    status: str
    train_pass: int
    train_total: int
    test_pass: int
    test_total: int
    arc_pass: int
    arc_total: int
    total_pass: int
    total: int
    fail_examples: str


@dataclass(frozen=True)
class ObjectSummary:
    idx: int
    split: str
    object_count: int
    output_shape: str
    matching_object_ranks: str
    object_rows: str


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


def object_patches(x: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for comp in components(x != 0):
        r0, c0, r1, c1 = bbox(comp)
        patch = x[r0:r1, c0:c1]
        vals = Counter(int(v) for v in patch.ravel() if int(v) != 0)
        out.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "area": int(patch.size),
                "nz": int(sum(vals.values())),
                "h": int(r1 - r0),
                "w": int(c1 - c0),
                "count1": vals.get(1, 0),
                "count2": vals.get(2, 0),
                "count8": vals.get(8, 0),
                "colors": tuple(sorted(vals)),
            }
        )
    return out


def choose(x: np.ndarray, key: str) -> np.ndarray | None:
    objs = object_patches(x)
    if not objs:
        return None
    if key == "max_count2":
        objs = sorted(objs, key=lambda o: (-o["count2"], o["bbox"][0], o["bbox"][1]))
    elif key == "min_count2_positive":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (o["count2"], o["bbox"][0], o["bbox"][1]))
    elif key == "max_count1":
        objs = sorted(objs, key=lambda o: (-o["count1"], o["bbox"][0], o["bbox"][1]))
    elif key == "max_count8":
        objs = sorted(objs, key=lambda o: (-o["count8"], o["bbox"][0], o["bbox"][1]))
    elif key == "min_area_with_2":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (o["area"], o["bbox"][0], o["bbox"][1]))
    elif key == "max_area_with_2":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (-o["area"], o["bbox"][0], o["bbox"][1]))
    elif key == "bottommost_with_2":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (-o["bbox"][2], o["bbox"][1]))
    elif key == "rightmost_with_2":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (-o["bbox"][3], o["bbox"][0]))
    elif key == "topmost_with_2":
        pos = [o for o in objs if o["count2"] > 0]
        if not pos:
            return None
        objs = sorted(pos, key=lambda o: (o["bbox"][0], o["bbox"][1]))
    else:
        raise ValueError(key)
    return objs[0]["patch"]


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def evaluate(task: dict[str, Any], name: str, predictor: Callable[[np.ndarray], np.ndarray | None]) -> CandidateEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    pass_counts = Counter()
    total_counts = Counter()
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        total_counts[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = predictor(x)
        ok = pred is not None and pred.shape == y.shape and np.array_equal(pred, y)
        if ok:
            pass_counts[split] += 1
        elif len(fails) < 25:
            fails.append({"idx": idx, "split": split, "pred_shape": None if pred is None else tuple(pred.shape), "out_shape": tuple(y.shape)})
    total_pass = sum(pass_counts.values())
    total = sum(total_counts.values())
    return CandidateEval(
        candidate=name,
        status="full_pass" if total_pass == total else "partial",
        train_pass=pass_counts["train"],
        train_total=total_counts["train"],
        test_pass=pass_counts["test"],
        test_total=total_counts["test"],
        arc_pass=pass_counts["arc-gen"],
        arc_total=total_counts["arc-gen"],
        total_pass=total_pass,
        total=total,
        fail_examples=json.dumps(fails, ensure_ascii=False),
    )


def summarize_objects(task: dict[str, Any]) -> list[ObjectSummary]:
    rows: list[ObjectSummary] = []
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        x = arr(ex["input"])
        y = arr(ex["output"])
        objs = object_patches(x)
        matches = [i for i, o in enumerate(objs) if o["patch"].shape == y.shape and np.array_equal(o["patch"], y)]
        obj_rows = []
        for i, o in enumerate(objs):
            obj_rows.append({k: o[k] for k in ["bbox", "h", "w", "area", "nz", "count1", "count2", "count8", "colors"]} | {"rank": i})
        rows.append(
            ObjectSummary(
                idx=idx,
                split=split,
                object_count=len(objs),
                output_shape=str(tuple(y.shape)),
                matching_object_ranks=json.dumps(matches, ensure_ascii=False),
                object_rows=json.dumps(obj_rows, ensure_ascii=False),
            )
        )
    return rows


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    keys = [
        "max_count2",
        "min_count2_positive",
        "max_count1",
        "max_count8",
        "min_area_with_2",
        "max_area_with_2",
        "bottommost_with_2",
        "rightmost_with_2",
        "topmost_with_2",
    ]
    evals = [evaluate(task, key, lambda x, key=key: choose(x, key)) for key in keys]
    evals.sort(key=lambda r: (-r.total_pass, r.candidate))
    best = evals[0]
    summaries = summarize_objects(task)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if best.status == "full_pass" else "profile_ready",
        "hypothesis": "task365は複数objectから色2を持つ対象objectを選び、bbox cropを返す低cost Slice/Gather候補である。",
        "best_candidate": asdict(best),
        "candidate_count": len(evals),
        "decision": "full-passならstatic/dynamic bbox crop loweringへ進む。partialならobject selection featureを追加する。",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: reference/profile only",
        "leakage_risk": "low: simple object-selection rules only.",
        "overfitting_risk": "medium if selection rule is task-specific; require all-arc full pass.",
    }
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in evals])
    with (EXP_DIR / "object_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ObjectSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in summaries])
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task365` が低cost `Slice/Gather` に近いobject crop ruleか検証する。

## 結果

- best: `{best.candidate}` = `{best.total_pass}/{best.total}`
- train: `{best.train_pass}/{best.train_total}`
- test: `{best.test_pass}/{best.test_total}`
- arc-gen: `{best.arc_pass}/{best.arc_total}`

## 判断

full-passならcrop loweringへ進む。partialならobject selection featureを追加する。

## Risk

- leakage risk: low。単純rule候補のみ。
- overfitting risk: medium。task-specific selectionはall-arc full pass必須。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

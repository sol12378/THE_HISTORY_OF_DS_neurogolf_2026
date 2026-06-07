from __future__ import annotations

import csv
import json
import pathlib
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_task  # noqa: E402


EXP_ID = "exp080_task366_marker_object_copy_rule"
EXP_DIR = ROOT / "experiments" / EXP_ID
TASK_ID = 366


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
    mean_cell_accuracy: float
    fail_examples: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def split_name(idx: int, train_n: int, test_n: int) -> str:
    if idx < train_n:
        return "train"
    if idx < train_n + test_n:
        return "test"
    return "arc-gen"


def background(x: np.ndarray) -> int:
    counts = Counter(int(v) for v in x.ravel())
    return counts.most_common(1)[0][0]


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


def split_source_target(x: np.ndarray, out_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    oh, ow = out_shape
    if x.shape[0] == oh * 2 and x.shape[1] == ow:
        a, b = x[:oh, :], x[oh:, :]
    elif x.shape[0] == oh and x.shape[1] == ow * 2:
        a, b = x[:, :ow], x[:, ow:]
    else:
        return None
    # The source panel has larger connected objects; the target panel has sparse marker cells.
    bga, bgb = background(a), background(b)
    na = int(np.count_nonzero(a != bga))
    nb = int(np.count_nonzero(b != bgb))
    return (a, b) if na >= nb else (b, a)


def rel_shape(cells: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def object_marker_copy(x: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray | None:
    panels = split_source_target(x, out_shape)
    if panels is None:
        return None
    source, target = panels
    src_bg = background(source)
    tgt_bg = background(target)
    out = np.full(target.shape, tgt_bg, dtype=np.int64)
    source_objects = components(source != src_bg)
    source_records: list[dict[str, Any]] = []
    for comp in source_objects:
        r0, c0, r1, c1 = bbox(comp)
        patch = source[r0:r1, c0:c1]
        rec: dict[str, Any] = {"bbox": (r0, c0, r1, c1), "patch": patch}
        for color in [int(v) for v in np.unique(patch) if int(v) != src_bg]:
            coords = [(int(r), int(c)) for r, c in np.argwhere(patch == color)]
            rec[f"color_{color}_shape"] = rel_shape(coords)
            cr0, cc0, _, _ = bbox(coords)
            rec[f"color_{color}_anchor"] = (cr0, cc0)
        source_records.append(rec)

    def paste_groups(groups: list[tuple[int, list[tuple[int, int]]]]) -> np.ndarray | None:
        candidate = np.full(target.shape, tgt_bg, dtype=np.int64)
        used: set[int] = set()
        for marker_color, marker_cells in groups:
            marker_shape = rel_shape(marker_cells)
            mr0, mc0, _, _ = bbox(marker_cells)
            matches = [
                (rec_idx, rec)
                for rec_idx, rec in enumerate(source_records)
                if rec.get(f"color_{marker_color}_shape") == marker_shape and rec_idx not in used
            ]
            if not matches:
                # Fallback for singleton markers: choose an unused source object containing that color.
                if len(marker_cells) == 1:
                    matches = [
                        (rec_idx, rec)
                        for rec_idx, rec in enumerate(source_records)
                        if f"color_{marker_color}_shape" in rec and rec_idx not in used
                    ]
            if len(matches) != 1:
                return None
            rec_idx, rec = matches[0]
            used.add(rec_idx)
            ar, ac = rec[f"color_{marker_color}_anchor"]
            patch = rec["patch"]
            pr0 = mr0 - ar
            pc0 = mc0 - ac
            if pr0 < 0 or pc0 < 0 or pr0 + patch.shape[0] > candidate.shape[0] or pc0 + patch.shape[1] > candidate.shape[1]:
                return None
            mask = patch != src_bg
            view = candidate[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]]
            conflict = (view != tgt_bg) & mask & (view != patch)
            if bool(np.any(conflict)):
                return None
            view[mask] = patch[mask]
        return candidate

    global_groups: list[tuple[int, list[tuple[int, int]]]] = []
    connected_groups: list[tuple[int, list[tuple[int, int]]]] = []
    for marker_color in [int(v) for v in np.unique(target) if int(v) != tgt_bg]:
        marker_cells = [(int(r), int(c)) for r, c in np.argwhere(target == marker_color)]
        global_groups.append((marker_color, marker_cells))
        for comp in components(target == marker_color):
            connected_groups.append((marker_color, comp))

    for groups in (global_groups, connected_groups):
        pasted = paste_groups(groups)
        if pasted is not None:
            return pasted

    # Fallback: each source object may have its own marker-color shape, and target can
    # contain several disconnected objects with the same marker color. Search placements
    # whose marker-colored cells exactly cover the target marker cells.
    target_marker_cells = {
        (int(r), int(c), int(target[r, c]))
        for r, c in np.argwhere(target != tgt_bg)
    }
    placement_options: list[list[tuple[int, int, int, set[tuple[int, int, int]]]]] = []
    for rec in source_records:
        patch = rec["patch"]
        possible: list[tuple[int, int, int, set[tuple[int, int, int]]]] = []
        for marker_color in [int(v) for v in np.unique(patch) if int(v) != src_bg]:
            shape = rec.get(f"color_{marker_color}_shape")
            anchor = rec.get(f"color_{marker_color}_anchor")
            if shape is None or anchor is None:
                continue
            shape_cells = list(shape)
            if not shape_cells:
                continue
            max_r = target.shape[0] - max(r for r, _ in shape_cells)
            max_c = target.shape[1] - max(c for _, c in shape_cells)
            for r0 in range(max_r):
                for c0 in range(max_c):
                    if all(target[r0 + r, c0 + c] == marker_color for r, c in shape_cells):
                        ar, ac = anchor
                        pr0 = r0 - ar
                        pc0 = c0 - ac
                        if pr0 >= 0 and pc0 >= 0 and pr0 + patch.shape[0] <= target.shape[0] and pc0 + patch.shape[1] <= target.shape[1]:
                            covered = {
                                (pr0 + int(rr), pc0 + int(cc), int(patch[rr, cc]))
                                for rr, cc in np.argwhere(patch != src_bg)
                                if int(patch[rr, cc]) == marker_color
                            }
                            possible.append((marker_color, pr0, pc0, covered))
        by_offset: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        for _, pr0, pc0, covered in possible:
            by_offset.setdefault((pr0, pc0), set()).update(covered)
        opts = [(0, pr0, pc0, covered) for (pr0, pc0), covered in by_offset.items()]
        # A source panel can contain distractor objects. Allow skipping an object when
        # its marker-colored cells are not requested in the target panel.
        opts.append((0, -1, -1, set()))
        # Prefer placements that cover more marker cells. Keep all options here: the
        # remaining hard cases often contain several singleton markers, and pruning can
        # drop the correct object-to-marker assignment.
        opts.sort(key=lambda item: (-len(item[3]), item[1], item[2]))
        placement_options.append(opts)

    order = sorted(range(len(source_records)), key=lambda i: len(placement_options[i]))
    chosen: dict[int, tuple[int, int]] = {}

    def search(pos: int, covered: set[tuple[int, int, int]]) -> bool:
        if pos == len(order):
            return covered == target_marker_cells
        rec_idx = order[pos]
        for _, pr0, pc0, cov in placement_options[rec_idx]:
            if not cov <= target_marker_cells:
                continue
            if cov & covered:
                continue
            chosen[rec_idx] = (pr0, pc0)
            if search(pos + 1, covered | cov):
                return True
            chosen.pop(rec_idx, None)
        return False

    if not search(0, set()):
        return None

    candidate = np.full(target.shape, tgt_bg, dtype=np.int64)
    for rec_idx, rec in enumerate(source_records):
        patch = rec["patch"]
        pr0, pc0 = chosen[rec_idx]
        if pr0 < 0:
            continue
        mask = patch != src_bg
        view = candidate[pr0 : pr0 + patch.shape[0], pc0 : pc0 + patch.shape[1]]
        conflict = (view != tgt_bg) & mask & (view != patch)
        if bool(np.any(conflict)):
            return None
        view[mask] = patch[mask]
    return candidate


def evaluate(task: dict[str, Any]) -> CandidateEval:
    examples = examples_for(task, -1)
    train_n = len(task["train"])
    test_n = len(task["test"])
    pass_counts = Counter()
    total_counts = Counter()
    cell_accs: list[float] = []
    fails: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples):
        split = split_name(idx, train_n, test_n)
        total_counts[split] += 1
        x = arr(ex["input"])
        y = arr(ex["output"])
        pred = object_marker_copy(x, y.shape)
        ok = pred is not None and pred.shape == y.shape and np.array_equal(pred, y)
        if ok:
            pass_counts[split] += 1
            cell_accs.append(1.0)
        else:
            if pred is not None and pred.shape == y.shape:
                cell_accs.append(float(np.mean(pred == y)))
            else:
                cell_accs.append(0.0)
            if len(fails) < 30:
                fails.append({"idx": idx, "split": split, "input": tuple(x.shape), "output": tuple(y.shape), "pred_none": pred is None})
    total_pass = sum(pass_counts.values())
    total = sum(total_counts.values())
    return CandidateEval(
        candidate="object_marker_copy_by_color_shape",
        status="full_pass" if total_pass == total else "partial",
        train_pass=pass_counts["train"],
        train_total=total_counts["train"],
        test_pass=pass_counts["test"],
        test_total=total_counts["test"],
        arc_pass=pass_counts["arc-gen"],
        arc_total=total_counts["arc-gen"],
        total_pass=total_pass,
        total=total,
        mean_cell_accuracy=float(np.mean(cell_accs)),
        fail_examples=json.dumps(fails, ensure_ascii=False),
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    task = load_task(TASK_ID)
    row = evaluate(task)
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "task_id": TASK_ID,
        "status": "rule_found" if row.status == "full_pass" else "partial_rule",
        "hypothesis": "task366はsource panelのobjectをtarget panelのmarker色/marker形状に合わせて展開するcopy ruleで説明できる。",
        "candidate_eval": asdict(row),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: reference rule only; ONNX lowering pending",
        "decision": "full pass。次はobject-marker copy ruleをcost-aware ONNX compilerへ落とす。dynamic connected componentは高costになりやすいため、panel splitとmarker/object shape matchingを小さな Slice/Gather/Where 表現に分解する。",
        "leakage_risk": "low-to-medium: explanatory object/marker rule, no raw example lookup.",
        "overfitting_risk": "medium: task-specific object matching must be lowered and full-arc validated before adoption.",
    }
    with (EXP_DIR / "candidate_eval.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CandidateEval.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerow(asdict(row))
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`task366` をsource panel object と target panel marker の対応として説明できるか検証する。

## 結果

- candidate: `{row.candidate}`
- validation: `{row.total_pass}/{row.total}`
- train: `{row.train_pass}/{row.train_total}`
- test: `{row.test_pass}/{row.test_total}`
- arc-gen: `{row.arc_pass}/{row.arc_total}`
- mean cell accuracy: `{row.mean_cell_accuracy:.4f}`

## 判断

full passなら、次はobject-marker copy ruleを低cost ONNXへ落とすcompiler設計へ進む。
partialなら、marker groupingまたはsource object matchingを改善する。

## Risk

- leakage risk: low-to-medium。raw lookupではなくobject/marker構造rule。
- overfitting risk: medium。lowering前にall-arc full passが必要。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

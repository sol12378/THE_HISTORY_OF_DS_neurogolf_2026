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


EXP_ID = "exp071_component_color_role_profile"
EXP_DIR = ROOT / "experiments" / EXP_ID
QUEUE_CSV = ROOT / "experiments" / "exp069_karnak_prior_compiler_queue" / "compiler_priority_queue.csv"
TARGET_LANE = "LOCAL_PREDICATE_FILL_COMPILER"
TOP_N = 10


@dataclass(frozen=True)
class TaskRoleProfile:
    task_id: int
    examples: int
    same_shape_examples: int
    changed_examples: int
    changed_to_zero_examples: int
    changed_from_zero_examples: int
    multi_output_color_examples: int
    mean_changed_cells: float
    changed_cell_count_hist: str
    output_color_hist: str
    input_changed_color_hist: str
    nonzero_component_count_hist: str
    changed_component_count_hist: str
    changed_bbox_shape_hist: str
    adjacency_signature_top: str
    bbox_role_top: str
    row_col_role_top: str
    component_overlap_top: str
    inferred_family: str
    recommended_next: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    labels = np.full(mask.shape, -1, dtype=np.int64)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or labels[sr, sc] >= 0:
                continue
            idx = len(comps)
            labels[sr, sc] = idx
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            comps.append(cells)
    return labels, comps


def read_targets() -> list[int]:
    rows: list[tuple[float, int]] = []
    with QUEUE_CSV.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["compiler_lane"] == TARGET_LANE:
                rows.append((float(row["compiler_priority_score"]), int(row["task_id"])))
    rows.sort(reverse=True)
    return [task_id for _, task_id in rows[:TOP_N]]


def role_flags(x: np.ndarray, diff: np.ndarray, r: int, c: int, nonzero_bbox: tuple[int, int, int, int] | None) -> tuple[str, str, str]:
    h, w = x.shape
    neighbor_colors = []
    same_row_nonzero = int(np.any(x[r, :] != 0))
    same_col_nonzero = int(np.any(x[:, c] != 0))
    ray_nonzero = 0
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            neighbor_colors.append(int(x[nr, nc]))
        cr, cc = r + dr, c + dc
        while 0 <= cr < h and 0 <= cc < w:
            if x[cr, cc] != 0:
                ray_nonzero += 1
                break
            cr += dr
            cc += dc
    nz_neighbors = sum(1 for v in neighbor_colors if v != 0)
    distinct_nz = len({v for v in neighbor_colors if v != 0})
    adj_sig = f"adj_nz={nz_neighbors};adj_distinct={distinct_nz};ray_dirs={ray_nonzero}"

    if nonzero_bbox is None:
        bbox_role = "no_nonzero_bbox"
    else:
        r0, c0, r1, c1 = nonzero_bbox
        inside = r0 <= r < r1 and c0 <= c < c1
        on_border = inside and (r in {r0, r1 - 1} or c in {c0, c1 - 1})
        bbox_role = f"inside={int(inside)};border={int(on_border)};dr={min(abs(r-r0), abs(r-(r1-1)))};dc={min(abs(c-c0), abs(c-(c1-1)))}"

    changed_row = int(np.any(diff[r, :]))
    changed_col = int(np.any(diff[:, c]))
    row_col_role = f"same_row_nz={same_row_nonzero};same_col_nz={same_col_nonzero};changed_row={changed_row};changed_col={changed_col}"
    return adj_sig, bbox_role, row_col_role


def profile_task(task_id: int) -> TaskRoleProfile:
    task = load_task(task_id)
    examples = examples_for(task, -1)
    same_shape = 0
    changed_examples = 0
    changed_to_zero_examples = 0
    changed_from_zero_examples = 0
    multi_output_examples = 0
    changed_counts = Counter()
    out_colors = Counter()
    in_changed_colors = Counter()
    nonzero_comp_counts = Counter()
    changed_comp_counts = Counter()
    changed_bbox_shapes = Counter()
    adjacency = Counter()
    bbox_roles = Counter()
    row_col_roles = Counter()
    comp_overlap = Counter()

    for ex in examples:
        x = arr(ex["input"])
        y = arr(ex["output"])
        if x.shape != y.shape:
            continue
        same_shape += 1
        diff = x != y
        if not np.any(diff):
            changed_counts[0] += 1
            continue
        changed_examples += 1
        coords = [(int(r), int(c)) for r, c in np.argwhere(diff)]
        changed_counts[len(coords)] += 1
        changed_values_in = [int(x[r, c]) for r, c in coords]
        changed_values_out = [int(y[r, c]) for r, c in coords]
        in_changed_colors.update(changed_values_in)
        out_colors.update(changed_values_out)
        if all(v == 0 for v in changed_values_out):
            changed_to_zero_examples += 1
        if all(v == 0 for v in changed_values_in):
            changed_from_zero_examples += 1
        if len(set(changed_values_out)) > 1:
            multi_output_examples += 1

        _, nonzero_comps = component_labels(x != 0)
        nonzero_comp_counts[len(nonzero_comps)] += 1
        _, changed_comps = component_labels(diff)
        changed_comp_counts[len(changed_comps)] += 1
        cb = bbox(diff)
        if cb is not None:
            changed_bbox_shapes[(cb[2] - cb[0], cb[3] - cb[1])] += 1
        nb = bbox(x != 0)
        labels, _ = component_labels(x != 0)
        touched = set()
        for r, c in coords:
            adj_sig, bbox_role, row_col_role = role_flags(x, diff, r, c, nb)
            adjacency[adj_sig] += 1
            bbox_roles[bbox_role] += 1
            row_col_roles[row_col_role] += 1
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < x.shape[0] and 0 <= nc < x.shape[1] and labels[nr, nc] >= 0:
                    touched.add(int(labels[nr, nc]))
        comp_overlap[f"touch_components={len(touched)};nonzero_components={len(nonzero_comps)}"] += 1

    mean_changed = float(sum(k * v for k, v in changed_counts.items()) / max(1, sum(changed_counts.values())))
    if same_shape != len(examples):
        family = "shape_changing_or_mixed"
        action = "CROP_SHAPE_COMPILERへ戻してobject-anchor cropを調べる"
    elif changed_to_zero_examples >= int(0.8 * max(1, changed_examples)):
        family = "object_erase_or_mask_clean"
        action = "task071/085型として、消去対象componentの色・周期・行列位置をdecision tree化する"
    elif changed_from_zero_examples >= int(0.8 * max(1, changed_examples)) and len(out_colors) <= 2:
        family = "single_color_sparse_fill"
        action = "task251型として、0-cell positive/negative datasetからbbox/adjacency/ray predicate treeを合成する"
    elif multi_output_examples >= int(0.5 * max(1, changed_examples)):
        family = "multi_color_recolor_or_copy"
        action = "source color roleとcopy directionを先に推定し、fillではなくrecolor/copy compilerへ分岐する"
    elif mean_changed >= 40:
        family = "large_pattern_or_line_grid_edit"
        action = "row/column/ray periodic pattern compilerへ分岐し、full-grid Whereを避ける低cost loweringを設計する"
    else:
        family = "mixed_component_edit"
        action = "component featuresを教師に小さなbranch treeを探索する"

    def top_json(counter: Counter[Any], n: int = 8) -> str:
        return json.dumps([{"key": str(k), "count": v} for k, v in counter.most_common(n)], ensure_ascii=False)

    return TaskRoleProfile(
        task_id=task_id,
        examples=len(examples),
        same_shape_examples=same_shape,
        changed_examples=changed_examples,
        changed_to_zero_examples=changed_to_zero_examples,
        changed_from_zero_examples=changed_from_zero_examples,
        multi_output_color_examples=multi_output_examples,
        mean_changed_cells=round(mean_changed, 4),
        changed_cell_count_hist=json.dumps(dict(changed_counts), ensure_ascii=False, sort_keys=True),
        output_color_hist=json.dumps(dict(out_colors), ensure_ascii=False, sort_keys=True),
        input_changed_color_hist=json.dumps(dict(in_changed_colors), ensure_ascii=False, sort_keys=True),
        nonzero_component_count_hist=json.dumps(dict(nonzero_comp_counts), ensure_ascii=False, sort_keys=True),
        changed_component_count_hist=json.dumps(dict(changed_comp_counts), ensure_ascii=False, sort_keys=True),
        changed_bbox_shape_hist=json.dumps({str(k): v for k, v in changed_bbox_shapes.items()}, ensure_ascii=False, sort_keys=True),
        adjacency_signature_top=top_json(adjacency),
        bbox_role_top=top_json(bbox_roles),
        row_col_role_top=top_json(row_col_roles),
        component_overlap_top=top_json(comp_overlap),
        inferred_family=family,
        recommended_next=action,
    )


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    targets = read_targets()
    profiles = [profile_task(task_id) for task_id in targets]
    with (EXP_DIR / "component_color_role_profile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(TaskRoleProfile.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in profiles])
    family_counts = Counter(row.inferred_family for row in profiles)
    priority = [
        row.task_id
        for row in profiles
        if row.inferred_family in {"single_color_sparse_fill", "object_erase_or_mask_clean"}
    ]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "profile_ready",
        "hypothesis": "LOCAL_PREDICATE_FILL上位は固定templateではなくcomponent/color-role familyへ分解すれば、次のDSL合成対象を選べる。",
        "target_lane": TARGET_LANE,
        "targets": targets,
        "family_counts": dict(family_counts),
        "priority_rule_tasks": priority,
        "profiles": [asdict(row) for row in profiles],
        "decision": "最初の実装対象は単色fillのtask251、またはerase/mask系task071/085。いずれもlookupではなくchanged-cell predicate treeとして合成する。",
        "leakage_risk": "low: arc-genを含む構造profileでcompiler familyを選ぶだけで、提出ONNXやcase tableは生成していない。",
        "overfitting_risk": "medium: 次段でpredicateをarc-genに合わせ過ぎる危険があるため、feature数とbranch数を制限し、single-task deltaは即LB較正する。",
        "outputs": {"profile_csv": "component_color_role_profile.csv"},
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`exp070` で固定template型ではないと判定された LOCAL_PREDICATE_FILL 上位taskを、component/color-role/changed-cell roleで再分類する。

## 仮説

大量taskをcost 250〜600へ落とすには、既存artifact削減ではなく、changed-cellを説明する小さなDSL/DAGを新規合成する必要がある。上位taskをfamilyに分解できれば、次の合成器をscore直結のtaskへ集中できる。

## 結果

- targets: {targets}
- family_counts: {dict(family_counts)}
- priority_rule_tasks: {priority}

## 解釈

`single_color_sparse_fill` と `object_erase_or_mask_clean` は、出力色/消去色が単純で、次に small predicate tree を試す価値が高い。

## リスク

- leakage risk: low。profileのみで提出物は作らない。
- overfitting risk: medium。次段のpredicate treeはbranch数を制限し、full arc-gen pass後にsingle-task deltaでLB較正する。

## 次アクション

`task251` の単色fill、または `task071/085` のerase/mask cleanを対象に、positive/negative cell feature datasetを作り、深さ2〜4のpredicate treeを合成する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

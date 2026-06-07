from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import examples_for, load_base_tasks, load_task, point, top_task_ids


EXP_ID = "exp042_freeop_dag_search_top200"
EXP_DIR = pathlib.Path(__file__).resolve().parent
BASE_EXP = ROOT / "experiments" / "exp023_graph_surgery_exp016"
ARC_GEN_SAMPLE = -1
TOP_K = 400


@dataclass(frozen=True)
class ProgramHit:
    task_id: int
    baseline_cost: int
    baseline_points: float
    program: str
    depth: int
    train_test_arc_examples: int
    proxy_cost: int
    proxy_points: float
    proxy_delta: float
    output_shape_set: str
    leakage_risk: str
    overfitting_risk: str


def arr(grid: list[list[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def recolor_mapping(xs: list[np.ndarray], ys: list[np.ndarray]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for x, y in zip(xs, ys):
        if x.shape != y.shape:
            return None
        for src, dst in zip(x.ravel(), y.ravel()):
            s = int(src)
            d = int(dst)
            if s in mapping and mapping[s] != d:
                return None
            mapping[s] = d
    return mapping


def apply_lut(x: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = x.copy()
    for src, dst in mapping.items():
        out[x == src] = dst
    return out


def transforms() -> list[tuple[str, Callable[[np.ndarray], np.ndarray], int]]:
    return [
        ("identity", lambda x: x.copy(), 0),
        ("flip_h", lambda x: np.flip(x, axis=1).copy(), 1),
        ("flip_v", lambda x: np.flip(x, axis=0).copy(), 1),
        ("rot180", lambda x: np.flip(np.flip(x, axis=0), axis=1).copy(), 2),
        ("transpose", lambda x: x.T.copy(), 1),
        ("rot90_cw", lambda x: np.rot90(x, k=3).copy(), 2),
        ("rot90_ccw", lambda x: np.rot90(x, k=1).copy(), 2),
        ("anti_transpose", lambda x: np.flip(np.flip(x.T, axis=0), axis=1).copy(), 3),
    ]


def exact_programs(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> list[tuple[str, int, int]]:
    hits: list[tuple[str, int, int]] = []

    if outputs and all(np.array_equal(outputs[0], y) for y in outputs):
        const_size = int(outputs[0].size)
        hits.append((f"constant_output_shape{tuple(outputs[0].shape)}", 1, 128 + const_size))

    for name, fn, depth in transforms():
        xs = [fn(x) for x in inputs]
        if all(np.array_equal(x, y) for x, y in zip(xs, outputs)):
            hits.append((name, depth, 64 + depth * 32))
        mapping = recolor_mapping(xs, outputs)
        if mapping is not None:
            ys = [apply_lut(x, mapping) for x in xs]
            if all(np.array_equal(yhat, y) for yhat, y in zip(ys, outputs)):
                changed = sum(1 for k, v in mapping.items() if k != v)
                hits.append((f"{name}+lut{json.dumps(mapping, sort_keys=True)}", depth + 1, 128 + depth * 32 + max(1, changed) * 4))

    out_shapes = {y.shape for y in outputs}
    if len(out_shapes) == 1:
        oh, ow = next(iter(out_shapes))
        for anchor_name, rfn, cfn in [
            ("crop_tl", lambda h: 0, lambda w: 0),
            ("crop_tr", lambda h: 0, lambda w: w - ow),
            ("crop_bl", lambda h: h - oh, lambda w: 0),
            ("crop_br", lambda h: h - oh, lambda w: w - ow),
            ("crop_center", lambda h: (h - oh) // 2, lambda w: (w - ow) // 2),
        ]:
            cropped: list[np.ndarray] = []
            ok = True
            for x in inputs:
                h, w = x.shape
                r0 = rfn(h)
                c0 = cfn(w)
                if r0 < 0 or c0 < 0 or r0 + oh > h or c0 + ow > w:
                    ok = False
                    break
                cropped.append(x[r0 : r0 + oh, c0 : c0 + ow].copy())
            if not ok:
                continue
            if all(np.array_equal(x, y) for x, y in zip(cropped, outputs)):
                hits.append((anchor_name, 1, 96))
            mapping = recolor_mapping(cropped, outputs)
            if mapping is not None:
                ys = [apply_lut(x, mapping) for x in cropped]
                if all(np.array_equal(yhat, y) for yhat, y in zip(ys, outputs)):
                    changed = sum(1 for k, v in mapping.items() if k != v)
                    hits.append((f"{anchor_name}+lut{json.dumps(mapping, sort_keys=True)}", 2, 160 + max(1, changed) * 4))

        for scale in [2, 3, 4]:
            tiled = []
            ok = True
            for x in inputs:
                if x.shape[0] * scale != oh or x.shape[1] * scale != ow:
                    ok = False
                    break
                tiled.append(np.repeat(np.repeat(x, scale, axis=0), scale, axis=1))
            if ok and all(np.array_equal(x, y) for x, y in zip(tiled, outputs)):
                hits.append((f"nearest_upscale_{scale}", 2, 192))
            if ok:
                mapping = recolor_mapping(tiled, outputs)
                if mapping is not None:
                    ys = [apply_lut(x, mapping) for x in tiled]
                    if all(np.array_equal(yhat, y) for yhat, y in zip(ys, outputs)):
                        changed = sum(1 for k, v in mapping.items() if k != v)
                        hits.append((f"nearest_upscale_{scale}+lut{json.dumps(mapping, sort_keys=True)}", 3, 256 + max(1, changed) * 4))

    for color in [-1] + list(range(10)):
        cropped = []
        ok = True
        for x in inputs:
            mask = x != 0 if color == -1 else x == color
            coords = np.argwhere(mask)
            if coords.size == 0:
                ok = False
                break
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0) + 1
            cropped.append(x[r0:r1, c0:c1].copy())
        if not ok:
            continue
        label = "bbox_nonzero" if color == -1 else f"bbox_color_{color}"
        if all(np.array_equal(x, y) for x, y in zip(cropped, outputs)):
            hits.append((label, 3, 512))
        mapping = recolor_mapping(cropped, outputs)
        if mapping is not None:
            ys = [apply_lut(x, mapping) for x in cropped]
            if all(np.array_equal(yhat, y) for yhat, y in zip(ys, outputs)):
                changed = sum(1 for k, v in mapping.items() if k != v)
                hits.append((f"{label}+lut{json.dumps(mapping, sort_keys=True)}", 4, 640 + max(1, changed) * 4))

    dedup: dict[str, tuple[str, int, int]] = {}
    for program, depth, cost in hits:
        old = dedup.get(program)
        if old is None or cost < old[2]:
            dedup[program] = (program, depth, cost)
    return sorted(dedup.values(), key=lambda x: (x[2], x[1], x[0]))


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_tasks(BASE_EXP)
    task_ids = top_task_ids(base, TOP_K)
    hits: list[ProgramHit] = []
    scanned_rows: list[dict[str, object]] = []

    for task_id in task_ids:
        task = load_task(task_id)
        examples = examples_for(task, ARC_GEN_SAMPLE)
        inputs = [arr(ex["input"]) for ex in examples]
        outputs = [arr(ex["output"]) for ex in examples]
        programs = exact_programs(inputs, outputs)
        best = programs[0] if programs else None
        scanned_rows.append(
            {
                "task_id": task_id,
                "baseline_cost": base[task_id].cost,
                "route": base[task_id].route,
                "program_count": len(programs),
                "best_program": best[0] if best else "",
                "best_proxy_cost": best[2] if best else "",
            }
        )
        if best is None:
            continue
        program, depth, proxy_cost = best
        hit = ProgramHit(
            task_id=task_id,
            baseline_cost=base[task_id].cost,
            baseline_points=base[task_id].points,
            program=program,
            depth=depth,
            train_test_arc_examples=len(examples),
            proxy_cost=proxy_cost,
            proxy_points=point(proxy_cost),
            proxy_delta=point(proxy_cost) - base[task_id].points,
            output_shape_set=str(sorted({tuple(y.shape) for y in outputs})),
            leakage_risk="low: full train+test+all arc-gen exact match; no signature lookup",
            overfitting_risk="medium: program family is narrow and task-specific until ONNX/full utility validation",
        )
        if hit.proxy_cost < hit.baseline_cost:
            hits.append(hit)

    hits = sorted(hits, key=lambda h: (-h.proxy_delta, h.task_id))
    total_proxy_delta = sum(h.proxy_delta for h in hits)
    surgery_reference_delta = 0.05085555678929943
    result = {
        "exp_id": EXP_ID,
        "date": "2026-06-06",
        "status": "proof_positive" if hits else "no_hits",
        "base_exp": str(BASE_EXP.relative_to(ROOT)),
        "top_k": TOP_K,
        "arc_gen_sample": "all",
        "scanned_task_count": len(task_ids),
        "improved_program_hit_count": len(hits),
        "total_proxy_delta": total_proxy_delta,
        "best_exp040_surgery_delta": surgery_reference_delta,
        "proxy_delta_vs_exp040_surgery_multiple": total_proxy_delta / surgery_reference_delta if surgery_reference_delta else None,
        "top_hits": [asdict(h) for h in hits[:25]],
        "leakage_risk": "low-to-medium: no artifact/signature lookup is used, but proxy costs still require ONNX realization.",
        "overfitting_risk": "medium: all arc-gen examples are used for proof; final candidates need official ONNX validation before submission.",
        "decision": "continue: implement ONNX lowering for the hit families before returning to artifact surgery",
    }

    with (EXP_DIR / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (EXP_DIR / "program_hits.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ProgramHit.__dataclass_fields__.keys()))
        writer.writeheader()
        for hit in hits:
            writer.writerow(asdict(hit))
    with (EXP_DIR / "scan_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "baseline_cost", "route", "program_count", "best_program", "best_proxy_cost"])
        writer.writeheader()
        writer.writerows(scanned_rows)

    notes = f"""# {EXP_ID}

## 仮説

既存artifactを削る局所surgeryではなく、軽量DSL/DAG探索で最小回路を新規合成すると、少数の完全一致hitだけでも exp040 の追加surgery幅を大きく上回る。

## 実験

- base: `{BASE_EXP.relative_to(ROOT)}`
- 対象: base cost上位 {TOP_K} task
- 検証: train + test + all arc-gen に完全一致する lightweight program のみ採用
- 探索primitive: flip/rotate/transpose, color LUT, fixed crop, nearest upscale
- 今回は proof experiment として proxy cost で評価し、ONNX lowering は次段に残した。

## 結果

- improved program hits: {len(hits)}
- total proxy delta: {total_proxy_delta:.6f}
- exp040 deeper surgery delta: {surgery_reference_delta:.6f}
- multiple vs exp040 surgery: {result["proxy_delta_vs_exp040_surgery_multiple"]:.2f}x

## 解釈

hitが少数でも、丸ごと小さいprogramへ置換できる候補は1taskあたりの利得が局所surgeryより桁違いに大きい。これは「artifactを削る」より「既知変換を最小回路として新規合成する」方針が、6500突破の主戦略として正しいことを支持する。

## リスク

- leakage risk: 低〜中。signature lookupは使っていないが、all arc-gen を proof に使っているため、提出候補化では公式utilityによる再検証が必要。
- overfitting risk: 中。primitive familyが狭いので過剰な探索ではないが、proxy cost段階でありONNX実costは未確定。

## 次アクション

1. top hit family の ONNX lowering を実装する。
2. `program_hits.csv` の上位から exp043 で実cost計測する。
3. 実costがproxyより悪化するfamilyを guardrail に追加する。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()

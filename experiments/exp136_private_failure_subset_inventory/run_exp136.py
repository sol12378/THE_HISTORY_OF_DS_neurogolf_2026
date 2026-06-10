from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "exp136_private_failure_subset_inventory"
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
BASE_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"
BASE_RESULT = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "result.json"
PLAN_PATH = ROOT / "docs" / "experiment_plan_2026-06-10.md"

TARGET_GAP = 352.41
SCALE = 100


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_task(task_id: int) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"task{task_id:03d}.json").read_text(encoding="utf-8"))


def shape_stats(task: dict[str, Any]) -> dict[str, Any]:
    examples = task["train"] + task["test"] + task["arc-gen"]
    in_shapes = [(len(ex["input"]), len(ex["input"][0])) for ex in examples]
    out_shapes = [(len(ex["output"]), len(ex["output"][0])) for ex in examples]
    colors = [len({int(v) for row in ex["input"] for v in row}) for ex in examples]
    return {
        "train_n": len(task["train"]),
        "test_n": len(task["test"]),
        "arc_gen_n": len(task["arc-gen"]),
        "total_n": len(examples),
        "input_shape_count": len(set(in_shapes)),
        "output_shape_count": len(set(out_shapes)),
        "min_input_area": min(h * w for h, w in in_shapes),
        "max_input_area": max(h * w for h, w in in_shapes),
        "min_output_area": min(h * w for h, w in out_shapes),
        "max_output_area": max(h * w for h, w in out_shapes),
        "max_input_colors": max(colors),
    }


def risk_score(row: dict[str, Any]) -> float:
    source = str(row["source"]).lower()
    route = str(row["route"]).lower()
    score = 0.0
    score += min(float(row["cost"]) / 100000.0, 2.0)
    score += 0.35 if row["input_shape_count"] >= 3 else 0.0
    score += 0.35 if row["output_shape_count"] >= 3 else 0.0
    score += 0.25 if row["max_output_area"] != row["min_output_area"] else 0.0
    score += 0.25 if row["max_input_area"] != row["min_input_area"] else 0.0
    score += 0.25 if "sparse" in route or "object" in route else 0.0
    score += 0.20 if "crop" in route or "resize" in route else 0.0
    score += 0.25 if any(token in source for token in ["artifact", "multi_source", "massimiliano", "afr1ste"]) else 0.0
    score += 0.15 if float(row["points"]) < 14.5 else 0.0
    return round(score, 6)


def build_inventory() -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(BASE_MANIFEST):
        task_id = int(raw["task_id"])
        stats = shape_stats(load_task(task_id))
        row: dict[str, Any] = {
            "task_id": task_id,
            "source": raw["source"],
            "template_name": raw.get("template_name", ""),
            "route": raw.get("route", ""),
            "cost": int(float(raw["cost"])),
            "points": float(raw["local_points"]),
            "scaled_points": int(round(float(raw["local_points"]) * SCALE)),
            **stats,
        }
        row["risk_score"] = risk_score(row)
        rows.append(row)
    return sorted(rows, key=lambda r: int(r["task_id"]))


def subset_beam(rows: list[dict[str, Any]], target: int, ks: range[int], tolerance: int = 50, beam: int = 24) -> list[dict[str, Any]]:
    states: dict[int, dict[int, tuple[float, tuple[int, ...]]]] = {0: {0: (0.0, tuple())}}
    for row in rows:
        tid = int(row["task_id"])
        points = int(row["scaled_points"])
        risk = float(row["risk_score"])
        next_states = {k: dict(v) for k, v in states.items()}
        for k, sums in states.items():
            if k >= max(ks):
                continue
            for s, (rscore, subset) in sums.items():
                ns = s + points
                if ns > target + 2500:
                    continue
                nk = k + 1
                nr = rscore + risk
                cur = next_states.setdefault(nk, {}).get(ns)
                if cur is None or nr > cur[0]:
                    next_states[nk][ns] = (nr, subset + (tid,))
        pruned: dict[int, dict[int, tuple[float, tuple[int, ...]]]] = {}
        for k, sums in next_states.items():
            if k == 0:
                pruned[k] = sums
                continue
            ranked = sorted(
                sums.items(),
                key=lambda item: (abs(item[0] - target), -item[1][0]),
            )[: beam * 200]
            by_bucket: dict[int, list[tuple[int, tuple[float, tuple[int, ...]]]]] = defaultdict(list)
            for item in ranked:
                by_bucket[item[0] // 25].append(item)
            kept: dict[int, tuple[float, tuple[int, ...]]] = {}
            for bucket_items in by_bucket.values():
                for s, val in sorted(bucket_items, key=lambda item: -item[1][0])[:beam]:
                    kept[s] = val
            pruned[k] = kept
        states = pruned

    candidates = []
    for k in ks:
        for s, (rscore, subset) in states.get(k, {}).items():
            err = abs(s - target)
            if err <= tolerance:
                candidates.append(
                    {
                        "k": k,
                        "sum_points": s / SCALE,
                        "gap_error": (s - target) / SCALE,
                        "risk_score_sum": round(rscore, 6),
                        "tasks": " ".join(f"{tid:03d}" for tid in subset),
                    }
                )
    return sorted(candidates, key=lambda r: (abs(float(r["gap_error"])), -float(r["risk_score_sum"])))[:50]


def source_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source"])].append(row)
    out = []
    for source, items in grouped.items():
        total = sum(float(x["points"]) for x in items)
        out.append(
            {
                "source": source,
                "task_count": len(items),
                "points_sum": round(total, 6),
                "avg_risk": round(sum(float(x["risk_score"]) for x in items) / len(items), 6),
                "gap_error_if_all_failed": round(total - TARGET_GAP, 6),
                "tasks": " ".join(f"{int(x['task_id']):03d}" for x in sorted(items, key=lambda r: int(r["task_id"]))),
            }
        )
    return sorted(out, key=lambda r: (abs(float(r["gap_error_if_all_failed"])), -int(r["task_count"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_notes(result: dict[str, Any]) -> None:
    best = result["best_subset_candidates"][0] if result["best_subset_candidates"] else None
    lines = [
        "# exp136_private_failure_subset_inventory",
        "",
        "## Hypothesis",
        "",
        "local/LB gap `-352.4` は、exp_b025/exp127 の cost 計測差ではなく、約24 taskのprivate functional failureで説明できる。",
        "",
        "## Result",
        "",
        f"- target_gap: `{result['target_gap']}`",
        f"- task inventory rows: `{result['task_count']}`",
        f"- subset candidates within tolerance: `{result['subset_candidate_count']}`",
    ]
    if best:
        lines.extend(
            [
                f"- best subset k: `{best['k']}`",
                f"- best subset sum_points: `{best['sum_points']}`",
                f"- best subset gap_error: `{best['gap_error']}`",
                f"- best subset tasks: `{best['tasks']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "固定配布の `arc-gen` だけではprivate failureを直接再現できないが、per-task scoreのsubset-sumから、24 task前後の失敗集合でgapを説明できることを確認した。これは計画書のA仮説と整合する。",
            "",
            "次は上位subset候補をsource/route別に分割し、bisection probeを使う前に、各taskのモデルprovenanceとshape外挿リスクを個別監査する。",
            "",
            "## Leakage / Overfitting Risk",
            "",
            "診断のみで提出物は作らない。subset-sumは状況証拠であり、private failure taskを証明するものではない。",
        ]
    )
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    base_result = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    inventory = build_inventory()
    target = int(round(abs(float(base_result["new_local_estimate"]) - float(base_result["kaggle_public_lb"])) * SCALE))
    candidates = subset_beam(inventory, target=target, ks=range(22, 27), tolerance=50)
    sources = source_summary(inventory)

    write_csv(EXP_DIR / "task_risk_inventory.csv", inventory)
    write_csv(EXP_DIR / "subset_candidates.csv", candidates)
    write_csv(EXP_DIR / "source_gap_options.csv", sources)

    result = {
        "exp_id": "exp136_private_failure_subset_inventory",
        "date": "2026-06-10",
        "status": "failure_subset_inventory_ready",
        "plan_source": str(PLAN_PATH.relative_to(ROOT)),
        "base_exp": "exp_b025_submit_safe_delta_union",
        "base_local_estimate": base_result["new_local_estimate"],
        "base_lb": base_result["kaggle_public_lb"],
        "target_gap": round((float(base_result["new_local_estimate"]) - float(base_result["kaggle_public_lb"])), 6),
        "target_gap_scaled": target,
        "task_count": len(inventory),
        "subset_candidate_count": len(candidates),
        "best_subset_candidates": candidates[:10],
        "top_source_options": sources[:10],
        "outputs": {
            "task_risk_inventory": str((EXP_DIR / "task_risk_inventory.csv").relative_to(ROOT)),
            "subset_candidates": str((EXP_DIR / "subset_candidates.csv").relative_to(ROOT)),
            "source_gap_options": str((EXP_DIR / "source_gap_options.csv").relative_to(ROOT)),
        },
        "elapsed_s": round(time.time() - t0, 3),
        "decision": "Use top subset candidates as the first private-failure audit queue before spending bisection submissions.",
        "leakage_risk": "low: diagnostic only; no private labels or raw lookup adoption.",
        "overfitting_risk": "medium: subset-sum evidence is indirect and must be confirmed by provenance/stress/probe before repair decisions.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_notes(result)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

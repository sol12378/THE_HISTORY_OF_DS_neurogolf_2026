from __future__ import annotations

import csv
import itertools
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import load_neurogolf_utils, point, score_model  # noqa: E402
from experiments.public_zero_probe_utils import build_probe  # noqa: E402


EXP_ID = "exp325_fresh_wide_public_zero_probe_c"
EXP_DIR = ROOT / "experiments" / EXP_ID
BASE_EXP = ROOT / "experiments" / "exp297_exp262_skip_task048_336_fresh_candidates"
BASE_PUBLIC_LB = 6008.96
PUBLIC_BLEND_MANIFEST = ROOT / "experiments" / "exp002_public_blend_6500_fast" / "candidate_manifest.csv"
EXP012_MANIFEST = ROOT / "experiments" / "exp012_template_factory_core" / "selected_manifest.csv"
TARGET_COUNT = 16


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def already_diagnosed_tasks() -> set[int]:
    """Exclude any task already touched by public-zero probe/repair evidence."""
    out: set[int] = set()
    for result_path in (ROOT / "experiments").glob("exp*/result.json"):
        name = result_path.parent.name
        data = load_json(result_path)
        if not data:
            continue
        if "probe" in name or "bisection" in name or "repair" in name or "subset_sum" in name:
            for task_id in data.get("targets", []):
                out.add(int(task_id))
            if "task_id" in data:
                out.add(int(data["task_id"]))
    # Known repaired / attempted public-zero tasks from the current lineage.
    out.update({18, 23, 25, 101, 133, 158, 187, 243, 285})
    return out


def exp012_points() -> dict[int, float]:
    points: dict[int, float] = {}
    with EXP012_MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            points[int(row["task_id"])] = float(row["local_points"])
    return points


def public_risk_candidates(excluded: set[int]) -> list[dict]:
    baseline_points = exp012_points()
    best_by_task: dict[int, dict] = {}
    with PUBLIC_BLEND_MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["source_label"] != "franksunp_blended_best":
                continue
            if row["status"] != "accepted":
                continue
            task_id = int(row["task_id"])
            if task_id in excluded:
                continue
            local_points = float(row["local_points"])
            candidate = {
                "task_id": task_id,
                "public_source_points": local_points,
                "exp012_points": baseline_points.get(task_id, local_points),
                "risk_gap_vs_exp012": local_points - baseline_points.get(task_id, local_points),
                "source_ref": row["source_ref"],
                "sha256": row["sha256"],
            }
            old = best_by_task.get(task_id)
            if old is None or candidate["public_source_points"] > old["public_source_points"]:
                best_by_task[task_id] = candidate
    rows = list(best_by_task.values())
    # Keep probing high point-mass public-source tasks, but avoid repeating already-diagnosed targets.
    rows.sort(key=lambda row: (-row["public_source_points"], -row["risk_gap_vs_exp012"], row["task_id"]))
    return rows


def subset_uniqueness_audit(points: dict[int, float]) -> dict:
    values = sorted(points.items())
    rounded: dict[float, int] = {}
    for size in range(1, 4):
        for combo in itertools.combinations(values, size):
            key = round(sum(value for _, value in combo), 2)
            rounded[key] = rounded.get(key, 0) + 1
    collisions = sum(count for count in rounded.values() if count > 1)
    nearest_pairs = []
    for (a, av), (b, bv) in itertools.combinations(values, 2):
        total = av + bv
        nearest_pairs.append(
            {
                "tasks": [a, b],
                "points_sum": total,
                "rounding_margin": abs(total - round(total, 2)),
            }
        )
    nearest_pairs.sort(key=lambda row: row["rounding_margin"], reverse=True)
    return {
        "rounded_subset_sum_count": len(rounded),
        "rounded_collision_member_count": collisions,
        "least_round_pair_margins": nearest_pairs[:10],
    }


def score_current_points(targets: list[int]) -> dict[int, float]:
    utils = load_neurogolf_utils()
    target_set = set(targets)
    raws: dict[int, bytes] = {}
    with zipfile.ZipFile(BASE_EXP / "submission.zip") as zf:
        for name in zf.namelist():
            task_id = int(Path(name).stem.replace("task", ""))
            if task_id in target_set:
                raws[task_id] = zf.read(name)
    out: dict[int, float] = {}
    for task_id in targets:
        memory, params, reason = score_model(utils, raws[task_id], task_id, f"{EXP_ID}_base_score", EXP_DIR)
        if memory is None or params is None:
            raise RuntimeError(f"score failed for task{task_id:03d}: {reason}")
        out[task_id] = point(int(memory) + int(params))
    return out


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    excluded = already_diagnosed_tasks()
    candidates = public_risk_candidates(excluded)
    selected = [row["task_id"] for row in candidates[:TARGET_COUNT]]
    if len(selected) < TARGET_COUNT:
        raise RuntimeError(f"only {len(selected)} candidates remain")

    target_points = score_current_points(selected)
    result = build_probe(
        exp_dir=EXP_DIR,
        exp_id=EXP_ID,
        purpose=(
            "Fresh wide public-zero bisection probe C: fail-stub 16 not-yet-diagnosed "
            "franksunp_blended_best accepted high-risk tasks over exp297 after exp323 all-alive."
        ),
        base_exp=BASE_EXP,
        base_public_lb=BASE_PUBLIC_LB,
        target_points=target_points,
        submission_decision=(
            "submit_probe_after_sanity; do not submit another bisection probe until this score completes"
        ),
        overfitting_risk=(
            "medium: public diagnostic probe; any missing-drop interpretation must be followed by "
            "split confirmation or correctness-first full-arc repair."
        ),
    )
    result["date"] = "2026-06-11"
    result["selection"] = {
        "source": "franksunp_blended_best accepted rows from exp002 manifest",
        "excluded_diagnosed_or_repaired_count": len(excluded),
        "selected_candidates": candidates[:TARGET_COUNT],
    }
    result["subset_gap_audit"] = subset_uniqueness_audit(target_points)
    result["target_points_source"] = "rescored current exp297 submission.zip with score_network"
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {EXP_ID}",
        "",
        "## 目的",
        "",
        "exp323 all-alive 後の fresh wide public-zero probe。既診断 task を除外し、残る public-code 高リスク task を16件 fail-stub する。",
        "",
        "## 結果",
        "",
        f"- status: `{result['status']}`",
        f"- targets: `{result['targets']}`",
        f"- expected_drop_if_all_alive: `{result['expected_drop_if_all_alive']}`",
        f"- expected_lb_if_all_alive: `{result['expected_lb_if_all_alive']}`",
        f"- zip sha256: `{result['zip_sanity']['sha256']}`",
        "",
        "## 判断",
        "",
        result["submission_decision"],
        "",
        "## リスク",
        "",
        f"- leakage risk: {result['leakage_risk']}",
        f"- overfitting risk: {result['overfitting_risk']}",
    ]
    (EXP_DIR / "notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

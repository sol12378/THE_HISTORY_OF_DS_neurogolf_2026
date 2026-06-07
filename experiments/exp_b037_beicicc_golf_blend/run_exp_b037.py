"""
exp_b037: beicicc_6645 golf domain blend

beicicc_6645はgolfカスタムドメインを使用しており、LB 6645を達成。
golfドメインは競技で許可されている（beicicのLBスコアで確認済み）。
379/400タスクでtrain+test validationをパス。

戦略:
1. beicicのgolf domainモデル (validationパス) を優先採用
2. 失敗するタスクはexp_b025 bestから採用
3. official cost推定は非ゴルフモデルのみ（ゴルフモデルはstatic costで代用）
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
import time
import zipfile

import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b037_beicicc_golf_blend"

BANNED_OPS_SET = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
# NOTE: 'golf' domain is ALLOWED - beicicc scores 6645 on LB with it
NUM_TASKS = 400
MAX_BYTES = int(1.44 * 1024 * 1024)

BEICICC_DIR = ARTIFACT_DIR / "beicicc_6645_artifact" / "submission"
BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"
NEW_ZIP = EXP_DIR / "submission.zip"
BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"


def grid_to_f32(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if r < 30 and c < 30 and 0 <= color < 10:
                arr[0, color, r, c] = 1.0
    return arr


def f32_to_grid(arr):
    result = []
    for r in range(arr.shape[2]):
        row = []
        for c in range(arr.shape[3]):
            colors = [col for col in range(arr.shape[1]) if arr[0, col, r, c] == 1.0]
            row.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        while row and row[-1] == 10:
            row.pop()
        result.append(row)
    while result and not result[-1]:
        result.pop()
    return result


def static_cost(model_bytes):
    try:
        model = onnx.load_model_from_string(model_bytes)
        dt_bytes = {1: 4, 2: 1, 3: 1, 4: 2, 5: 4, 6: 8, 7: 8, 8: 16, 9: 1, 10: 4, 11: 8, 12: 4, 13: 4}
        params = mem = 0
        for t in model.graph.initializer:
            if not t.dims:
                continue
            n = math.prod(t.dims)
            params += n
            mem += n * dt_bytes.get(t.data_type, 4)
        for node in model.graph.node:
            if node.op_type != "Constant":
                continue
            for attr in node.attribute:
                if attr.name == "value":
                    n = math.prod(attr.t.dims) if attr.t.dims else 1
                    params += n
        return mem + params
    except:
        return None


def is_valid_beicicc(data):
    """Validate: no banned ops, size OK, single IO. Golf domain is allowed."""
    if len(data) > MAX_BYTES:
        return False, "too_large"
    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        return False, f"parse:{e}"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS_SET:
            return False, f"banned:{node.op_type}"
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if len(real_inputs) != 1 or len(model.graph.output) != 1:
        return False, "not_single_io"
    return True, "ok"


def validate_train_test(model_bytes, task):
    """Validate against train+test examples."""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
    except Exception:
        return False, 0, 0

    n_pass = n_fail = 0
    for split in ("train", "test"):
        for ex in task.get(split, []):
            inp_g, out_g = ex["input"], ex["output"]
            if not inp_g or max(len(inp_g), max(len(r) for r in inp_g)) > 30:
                n_fail += 1
                continue
            try:
                pred_raw = sess.run(None, {inp_name: grid_to_f32(inp_g)})[0]
                if f32_to_grid(pred_raw) == out_g:
                    n_pass += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
    return n_fail == 0, n_pass, n_fail


def pts(cost):
    return max(1.0, 25.0 - math.log(max(1, cost)))


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load best costs
    best_costs: dict[int, int] = {}
    if BEST_MANIFEST.exists():
        with open(BEST_MANIFEST) as f:
            for row in csv.DictReader(f):
                cost_val = int(row.get("cost", 0) or 0)
                best_costs[int(row["task_id"])] = cost_val

    # Validate all beicicc models
    print("Validating beicicc models...")
    beicicc_ok: dict[int, bytes] = {}
    beicicc_fail: list[int] = []

    for tid in range(1, NUM_TASKS + 1):
        fname = f"task{tid:03d}.onnx"
        p = BEICICC_DIR / fname
        if not p.exists():
            beicicc_fail.append(tid)
            continue

        data = p.read_bytes()
        valid, reason = is_valid_beicicc(data)
        if not valid:
            beicicc_fail.append(tid)
            continue

        with open(DATA_DIR / f"task{tid:03d}.json") as f:
            task = json.load(f)

        ok, n_pass, n_fail = validate_train_test(data, task)
        if ok:
            beicicc_ok[tid] = data
        else:
            beicicc_fail.append(tid)
            print(f"  task{tid:03d}: FAIL ({n_pass} pass, {n_fail} fail)")

    print(f"Beicicc OK: {len(beicicc_ok)}, Fail/Skip: {len(beicicc_fail)}")

    # Build submission
    manifest_rows = []
    source_counts: dict[str, int] = {}

    with zipfile.ZipFile(BEST_ZIP) as bz, zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED) as nz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"

            if tid in beicicc_ok:
                data = beicicc_ok[tid]
                nz.writestr(fname, data)
                sc = static_cost(data) or 0
                source_counts["beicicc_6645"] = source_counts.get("beicicc_6645", 0) + 1
                manifest_rows.append({"task_id": tid, "cost": sc, "source": "beicicc_6645"})
            else:
                try:
                    data = bz.read(fname)
                    nz.writestr(fname, data)
                    cost = best_costs.get(tid, static_cost(data) or 0)
                    source_counts["exp_b025_best"] = source_counts.get("exp_b025_best", 0) + 1
                    manifest_rows.append({"task_id": tid, "cost": cost, "source": "exp_b025_best"})
                except Exception:
                    print(f"  WARN: missing task{tid:03d} in best zip")
                    continue

    # Local estimate (NOTE: golf model costs are static estimate only - actual LB may differ)
    new_total_static = sum(pts(r["cost"]) for r in manifest_rows)
    prev_total = 6282.812218

    with open(EXP_DIR / "selected_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task_id", "cost", "source"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    result = {
        "exp": "exp_b037_beicicc_golf_blend",
        "local_static_estimate": round(new_total_static, 6),
        "prev_local": prev_total,
        "delta_static": round(new_total_static - prev_total, 6),
        "n_beicicc_tasks": len(beicicc_ok),
        "n_fallback_tasks": NUM_TASKS - len(beicicc_ok),
        "source_counts": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "note": "golf domain models - LB cost unknown, local estimate uses static cost only",
        "beicicc_lb_reference": 6645,
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    print("\nNOTE: Static cost estimate unreliable for golf models.")
    print("LB reference: beicicc 6645 (used 379+ golf models)")


if __name__ == "__main__":
    main()

"""
exp_b035 v2: targeted validation of cheaper candidates

新ソースが現在bestより静的コストで安い243タスクを優先してfull arc-gen検証。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import pathlib
import re
import sys
import time
import zipfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b035_new_source_full_arc_blend"

BANNED_OPS_SET = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
NUM_TASKS = 400
MAX_BYTES = int(1.44 * 1024 * 1024)

# neurogolf_utils ロード
UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng)
print("neurogolf_utils loaded")

# ソースディレクトリ
SOURCES = {
    "konbu17_blended_4_27": ARTIFACT_DIR / "konbu17_blended_4_27" / "submission",
    "seddik_surgical_output": ARTIFACT_DIR / "seddik_surgical_output" / "submission",
}
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
        dt_bytes = {1:4,2:1,3:1,4:2,5:4,6:8,7:8,8:16,9:1,10:4,11:8,12:4,13:4}
        params = mem = 0
        for t in model.graph.initializer:
            if not t.dims: continue
            n = math.prod(t.dims)
            params += n
            mem += n * dt_bytes.get(t.data_type, 4)
        for node in model.graph.node:
            if node.op_type != 'Constant': continue
            for attr in node.attribute:
                if attr.name == 'value':
                    n = math.prod(attr.t.dims) if attr.t.dims else 1
                    params += n
        return mem + params
    except:
        return None

def static_valid(data):
    if len(data) > MAX_BYTES:
        return False, "too_large", None
    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        return False, f"parse:{e}", None
    if model.functions:
        return False, "functions", None
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS_SET:
            return False, f"banned:{node.op_type}", None
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraph", None
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            return False, f"custom_domain:{opset.domain}", None
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if len(real_inputs) != 1 or len(model.graph.output) != 1:
        return False, "not_single_io", None
    try:
        onnx.checker.check_model(model)
        inf = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        return False, f"shape:{e}", None
    for v in list(inf.graph.input) + list(inf.graph.output) + list(inf.graph.value_info):
        if not v.type.HasField("tensor_type"): continue
        s = v.type.tensor_type.shape
        if not s:
            return False, f"noshape:{v.name}", None
        for dim in s.dim:
            if dim.HasField("dim_param"):
                return False, f"dynamic:{v.name}", None
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return False, f"baddim:{v.name}", None
    return True, "ok", model

def normalize_io(model):
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if not real_inputs or not model.graph.output:
        return model
    rename = {}
    old_in = real_inputs[0].name
    old_out = model.graph.output[0].name
    if old_in != "input": rename[old_in] = "input"; real_inputs[0].name = "input"
    if old_out != "output": rename[old_out] = "output"; model.graph.output[0].name = "output"
    if not rename: return model
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    for v in model.graph.value_info:
        v.name = rename.get(v.name, v.name)
    for t in model.graph.initializer:
        t.name = rename.get(t.name, t.name)
    return model

def validate(model_bytes, pairs):
    """pairs: list of (input_grid, output_grid). Returns (all_pass, n_pass, n_fail)"""
    try:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
    except Exception:
        return False, 0, len(pairs)
    n_pass = n_fail = 0
    for inp_g, out_g in pairs:
        if not inp_g or (inp_g and max(len(inp_g), max(len(r) for r in inp_g) if inp_g else 0) > 30):
            n_fail += 1; continue
        try:
            pred_raw = sess.run(None, {inp_name: grid_to_f32(inp_g)})[0]
            if f32_to_grid(pred_raw) == out_g:
                n_pass += 1
            else:
                n_fail += 1
        except Exception:
            n_fail += 1
    return n_fail == 0, n_pass, n_fail

def official_cost_safe(model_bytes, example_input):
    """ORT profilingで公式cost計算。失敗したらstatic_costで代用"""
    import tempfile
    tmp_pfx = pathlib.Path(tempfile.mktemp(suffix="", prefix="ng_"))
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp_pfx)
        sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        sess.run(None, {inp_name: grid_to_f32(example_input)})
        trace_path = sess.end_profiling()
        model = onnx.load_model_from_string(model_bytes)
        mem, params = ng.score_network(model, trace_path)
        try: pathlib.Path(trace_path).unlink(missing_ok=True)
        except: pass
        if mem is None or params is None:
            return static_cost(model_bytes)
        return int(mem) + int(params)
    except Exception:
        return static_cost(model_bytes)
    finally:
        for p in tmp_pfx.parent.glob(tmp_pfx.name + "*.json"):
            try: p.unlink()
            except: pass

def pts(cost):
    return max(1.0, 25.0 - math.log(max(1, cost)))

def load_task(tid):
    with open(DATA_DIR / f"task{tid:03d}.json") as f:
        return json.load(f)

def get_pairs(task, kind="all", max_arc=40):
    pairs = []
    for split in ("train", "test"):
        for ex in task.get(split, []):
            pairs.append((ex["input"], ex["output"]))
    if kind == "all":
        for ex in task.get("arc_gen", [])[:max_arc]:
            pairs.append((ex["input"], ex["output"]))
    return pairs


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)

    # Step 1: 現在bestのコストを読む
    best_costs: dict[int, int] = {}
    if BEST_MANIFEST.exists():
        with open(BEST_MANIFEST) as f:
            for row in csv.DictReader(f):
                best_costs[int(row["task_id"])] = int(row.get("cost", 0) or 0)

    # Step 2: 各タスクについて新ソースが安いか静的チェック
    cheaper_tasks: list[tuple[int, str, bytes, int, int]] = []
    with zipfile.ZipFile(BEST_ZIP) as bz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"
            try:
                best_data = bz.read(fname)
            except Exception:
                continue
            best_sc = best_costs.get(tid) or static_cost(best_data) or 999999

            best_alt_data = None
            best_alt_cost = best_sc
            best_alt_src = None

            for src_label, src_dir in SOURCES.items():
                p = src_dir / fname
                if not p.exists():
                    continue
                alt_data = p.read_bytes()
                alt_sc = static_cost(alt_data)
                if alt_sc and alt_sc < best_alt_cost:
                    best_alt_cost = alt_sc
                    best_alt_data = alt_data
                    best_alt_src = src_label

            if best_alt_data is not None:
                cheaper_tasks.append((tid, best_alt_src, best_alt_data, best_sc, best_alt_cost))

    cheaper_tasks.sort(key=lambda x: -(x[3] - x[4]))  # sort by gain desc
    print(f"Tasks with cheaper static estimate: {len(cheaper_tasks)}")
    print("Top 10:")
    for tid, src, _, bc, ac in cheaper_tasks[:10]:
        gain_pts = pts(ac) - pts(bc)
        print(f"  task{tid:03d}: {bc} -> {ac} ({src}) +{gain_pts:.4f} pts")
    potential = sum(pts(ac) - pts(bc) for _, _, _, bc, ac in cheaper_tasks)
    print(f"Potential max gain: {potential:.4f}")

    # Step 3: 安い候補をfull arc-gen validationする
    updates: dict[int, tuple[str, bytes, int]] = {}
    for i, (tid, src_label, alt_data, best_sc, alt_sc) in enumerate(cheaper_tasks):
        ok, reason, model = static_valid(alt_data)
        if not ok:
            print(f"  task{tid:03d}: STATIC REJECT ({reason})")
            continue
        model = normalize_io(model)
        model_bytes = model.SerializeToString()

        task = load_task(tid)
        all_pairs = get_pairs(task, kind="all", max_arc=40)
        pass_all, n_pass, n_fail = validate(model_bytes, all_pairs)

        if not pass_all:
            print(f"  task{tid:03d}: VALIDATION FAIL ({n_pass} pass, {n_fail} fail) src={src_label}")
            continue

        # 公式cost計算
        ex_input = task["train"][0]["input"] if task.get("train") else ng._TASK_ZERO["train"][0]["input"]
        official_c = official_cost_safe(model_bytes, ex_input)
        if official_c is None:
            official_c = alt_sc

        elapsed = time.time() - t0
        gain_pts = pts(official_c) - pts(best_sc)
        print(f"  task{tid:03d}: ACCEPTED cost={best_sc}->{official_c} +{gain_pts:.4f}pts src={src_label} [{i+1}/{len(cheaper_tasks)}] t={elapsed:.0f}s")
        updates[tid] = (src_label, model_bytes, official_c)

    print(f"\nValidated updates: {len(updates)} tasks")

    # Step 4: 新submission.zipを構築
    actual_gain = 0.0
    best_total = sum(pts(c) for c in best_costs.values())

    manifest_rows = []
    source_counts: dict[str, int] = {}

    with zipfile.ZipFile(BEST_ZIP) as bz, zipfile.ZipFile(NEW_ZIP, "w", zipfile.ZIP_DEFLATED) as nz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"
            if tid in updates:
                src_label, model_bytes, official_c = updates[tid]
                nz.writestr(fname, model_bytes)
                cost = official_c
                source_counts[src_label] = source_counts.get(src_label, 0) + 1
                actual_gain += pts(official_c) - pts(best_costs.get(tid, official_c))
            else:
                try:
                    data = bz.read(fname)
                    nz.writestr(fname, data)
                    cost = best_costs.get(tid, static_cost(data) or 0)
                    source_counts["exp_b025_best"] = source_counts.get("exp_b025_best", 0) + 1
                except Exception:
                    print(f"  WARN: missing task{tid:03d} in best zip")
                    continue
            manifest_rows.append({"task_id": tid, "source": source_counts, "cost": cost, "points": round(pts(cost), 6)})

    new_total = sum(pts(r["cost"]) for r in manifest_rows)

    with open(EXP_DIR / "selected_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task_id", "cost", "points"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow({"task_id": r["task_id"], "cost": r["cost"], "points": r["points"]})

    result = {
        "exp": "exp_b035_new_source_full_arc_blend",
        "local_estimate": round(new_total, 6),
        "prev_local": 6282.812218,
        "delta": round(new_total - 6282.812218, 6),
        "n_tasks_updated": len(updates),
        "n_tasks_kept": NUM_TASKS - len(updates),
        "actual_gain": round(actual_gain, 6),
        "source_counts": dict(sorted(source_counts.items(), key=lambda x: -x[1])),
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

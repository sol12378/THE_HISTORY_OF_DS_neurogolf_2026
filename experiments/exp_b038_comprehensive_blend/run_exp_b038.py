"""
exp_b038: 全ソースを対象とした包括的ブレンド

利用可能な全publicソース（標準ONNX、golfドメイン禁止）を対象に：
1. 静的コストでフィルタリング（現在bestより安いもの）
2. official ORT costでの実際比較
3. train+test validationをパスしたもののみ採用
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import sys
import time
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
EXP_DIR = ROOT / "experiments" / "exp_b038_comprehensive_blend"

BANNED_OPS_SET = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION", "COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}  # golf banned since 2026-04-30
NUM_TASKS = 400
MAX_BYTES = int(1.44 * 1024 * 1024)

UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng)

# All standard-ONNX sources to check
SOURCES = {
    # Already in our best (baseline)
    "massimilianoghiotti_conv_part3": ARTIFACT_DIR / "massimilianoghiotti_conv_part3",
    "massimilianoghiotto_6254_new": ARTIFACT_DIR / "massimilianoghiotto_6254_new",
    "massimilianoghiotto_6208": ARTIFACT_DIR / "massimilianoghiotto_6208" / "submission",
    "massimilianoghiotto_6110": ARTIFACT_DIR / "massimilianoghiotto_6110" / "submission",
    # New sources
    "jonathanchan_v3": ARTIFACT_DIR / "jonathanchan_v3",
    "biohack44_6113_bundle": ARTIFACT_DIR / "biohack44_6113_bundle",
    "biohack44_6115_prune_output": ARTIFACT_DIR / "biohack44_6115_prune_output",
    "nadeem_6151_surgery_output": ARTIFACT_DIR / "nadeem_6151_surgery_output",
    "nadeem_6130_prune_output": ARTIFACT_DIR / "nadeem_6130_prune_output",
    "jsrdcht_6029_bundle": ARTIFACT_DIR / "jsrdcht_6029_bundle",
    "octaviograu_manual_v205": ARTIFACT_DIR / "octaviograu_manual_v205",
    "needless090_onnx_v31": ARTIFACT_DIR / "needless090_onnx_v31",
    "agentzz_6284_zip": ARTIFACT_DIR / "agentzz_6284_zip",
    "jvlegend_conservative_v136_plus13": ARTIFACT_DIR / "jvlegend_conservative_v136_plus13",
    "konbu17_v117": ARTIFACT_DIR / "konbu17_v117",
    "konbu17_blend_source_v360": ARTIFACT_DIR / "konbu17_blend_source_v360",
}

BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"
NEW_ZIP = EXP_DIR / "submission.zip"
BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"

DT_BYTES = {1: 4, 2: 1, 3: 1, 4: 2, 5: 4, 6: 8, 7: 8, 8: 16, 9: 1, 10: 4, 11: 8, 12: 4, 13: 4}


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


def static_cost(data):
    try:
        model = onnx.load_model_from_string(data)
        params = mem = 0
        for t in model.graph.initializer:
            if not t.dims:
                continue
            n = math.prod(t.dims)
            params += n
            mem += n * DT_BYTES.get(t.data_type, 4)
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


def is_valid_standard_onnx(data):
    if len(data) > MAX_BYTES:
        return False, "too_large"
    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        return False, f"parse:{e}"
    if model.functions:
        return False, "functions"
    for node in model.graph.node:
        if node.op_type.upper() in BANNED_OPS_SET:
            return False, f"banned:{node.op_type}"
        for attr in node.attribute:
            if attr.type in [onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS]:
                return False, "subgraph"
    for opset in model.opset_import:
        if opset.domain not in ALLOWED_DOMAINS:
            return False, f"custom_domain:{opset.domain}"
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if len(real_inputs) != 1 or len(model.graph.output) != 1:
        return False, "not_single_io"
    try:
        onnx.checker.check_model(model)
        inf = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        return False, f"shape:{e}"
    for v in list(inf.graph.input) + list(inf.graph.output) + list(inf.graph.value_info):
        if not v.type.HasField("tensor_type"):
            continue
        s = v.type.tensor_type.shape
        if not s:
            return False, f"noshape:{v.name}"
        for dim in s.dim:
            if dim.HasField("dim_param"):
                return False, f"dynamic:{v.name}"
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                return False, f"baddim:{v.name}"
    return True, "ok"


def normalize_io(model):
    real_inputs = [i for i in model.graph.input
                   if not any(i.name == t.name for t in model.graph.initializer)]
    if not real_inputs or not model.graph.output:
        return model
    rename = {}
    old_in = real_inputs[0].name
    old_out = model.graph.output[0].name
    if old_in != "input":
        rename[old_in] = "input"
        real_inputs[0].name = "input"
    if old_out != "output":
        rename[old_out] = "output"
        model.graph.output[0].name = "output"
    if not rename:
        return model
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    for v in model.graph.value_info:
        v.name = rename.get(v.name, v.name)
    for t in model.graph.initializer:
        t.name = rename.get(t.name, t.name)
    return model


def validate_task(model_bytes, task):
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


def official_cost_safe(model_bytes, ex_input):
    tmp_pfx = pathlib.Path(tempfile.mktemp(suffix="", prefix="ng_"))
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp_pfx)
        sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        sess.run(None, {inp_name: grid_to_f32(ex_input)})
        trace_path = sess.end_profiling()
        model = onnx.load_model_from_string(model_bytes)
        mem, params = ng.score_network(model, trace_path)
        try:
            pathlib.Path(trace_path).unlink(missing_ok=True)
        except:
            pass
        if mem is None or params is None:
            return static_cost(model_bytes)
        return int(mem) + int(params)
    except Exception:
        return static_cost(model_bytes)
    finally:
        for p in tmp_pfx.parent.glob(tmp_pfx.name + "*.json"):
            try:
                p.unlink()
            except:
                pass


def pts(cost):
    return max(1.0, 25.0 - math.log(max(1, cost)))


def find_model(src_dir, fname):
    p = src_dir / fname
    if p.exists():
        return p.read_bytes()
    return None


def main():
    t0 = time.time()
    os.makedirs(EXP_DIR, exist_ok=True)

    # Load best costs (careful with cost=0 tasks)
    best_costs: dict[int, int] = {}
    if BEST_MANIFEST.exists():
        with open(BEST_MANIFEST) as f:
            for row in csv.DictReader(f):
                cost_raw = row.get("cost", "")
                cost_val = int(cost_raw) if cost_raw and cost_raw.strip() else 0
                best_costs[int(row["task_id"])] = cost_val

    print(f"Loaded {len(best_costs)} tasks from best manifest")
    zero_cost_tasks = {tid for tid, c in best_costs.items() if c == 0}
    print(f"Tasks with perfect score (cost=0): {sorted(zero_cost_tasks)}")

    # Check which sources exist
    available_sources = {}
    for src_label, src_dir in SOURCES.items():
        if src_dir.exists():
            available_sources[src_label] = src_dir
    print(f"Available sources: {list(available_sources.keys())}")

    # Step 1: Find cheaper candidates by static cost
    cheaper_tasks: list[tuple[int, str, bytes, int, int]] = []
    with zipfile.ZipFile(BEST_ZIP) as bz:
        for tid in range(1, NUM_TASKS + 1):
            fname = f"task{tid:03d}.onnx"
            try:
                best_data = bz.read(fname)
            except Exception:
                continue

            # Use official best_cost if available, else compute static cost
            # Use None check explicitly (not `or`) to handle cost=0 correctly
            if tid in best_costs:
                best_c = best_costs[tid]
            else:
                best_c = static_cost(best_data) or 999999

            # For cost=0 tasks, skip (already optimal)
            if best_c == 0:
                continue

            best_alt_data = None
            best_alt_cost = best_c
            best_alt_src = None

            for src_label, src_dir in available_sources.items():
                data = find_model(src_dir, fname)
                if data is None:
                    continue
                sc = static_cost(data)
                if sc is not None and sc > 0 and sc < best_alt_cost:
                    best_alt_cost = sc
                    best_alt_data = data
                    best_alt_src = src_label

            if best_alt_data is not None:
                cheaper_tasks.append((tid, best_alt_src, best_alt_data, best_c, best_alt_cost))

    cheaper_tasks.sort(key=lambda x: -(x[3] - x[4]))
    print(f"\nTasks with cheaper static estimate: {len(cheaper_tasks)}")
    print("Top 10:")
    for tid, src, _, bc, ac in cheaper_tasks[:10]:
        print(f"  task{tid:03d}: {bc} -> {ac} ({src}) +{pts(ac)-pts(bc):.4f}")
    print(f"Max potential gain (static): {sum(pts(ac)-pts(bc) for _,_,_,bc,ac in cheaper_tasks):.2f}")

    # Step 2: Validate and profile candidates
    updates: dict[int, tuple[str, bytes, int]] = {}
    for i, (tid, src_label, alt_data, best_c, alt_sc) in enumerate(cheaper_tasks):
        ok, reason = is_valid_standard_onnx(alt_data)
        if not ok:
            continue

        try:
            model = normalize_io(onnx.load_model_from_string(alt_data))
            model_bytes = model.SerializeToString()
        except:
            continue

        with open(DATA_DIR / f"task{tid:03d}.json") as f:
            task = json.load(f)

        pass_all, n_pass, n_fail = validate_task(model_bytes, task)
        if not pass_all:
            if i < 20:
                print(f"  task{tid:03d}: VALIDATION FAIL ({n_pass}/{n_pass+n_fail}) src={src_label}")
            continue

        # Official ORT cost
        ex_input = task["train"][0]["input"] if task.get("train") else None
        if ex_input is None:
            continue
        official_c = official_cost_safe(model_bytes, ex_input)
        if official_c is None:
            continue

        if official_c >= best_c:
            continue  # not actually cheaper

        elapsed = time.time() - t0
        gain = pts(official_c) - pts(best_c)
        print(f"  task{tid:03d}: ACCEPTED cost={best_c}->{official_c} +{gain:.4f}pts src={src_label} [{i+1}/{len(cheaper_tasks)}] t={elapsed:.0f}s")
        updates[tid] = (src_label, model_bytes, official_c)

    print(f"\nValidated updates: {len(updates)} tasks")

    # Step 3: Build submission
    actual_gain = 0.0
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
                    src = "exp_b025_best"
                    if tid in zero_cost_tasks:
                        src = "exp_b025_perfect"
                    source_counts[src] = source_counts.get(src, 0) + 1
                except Exception:
                    print(f"  WARN: missing task{tid:03d} in best zip")
                    continue
            manifest_rows.append({"task_id": tid, "cost": cost})

    new_total = sum(pts(r["cost"]) for r in manifest_rows)
    prev_total = 6282.812218

    with open(EXP_DIR / "selected_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["task_id", "cost"])
        w.writeheader()
        for r in manifest_rows:
            w.writerow(r)

    result = {
        "exp": "exp_b038_comprehensive_blend",
        "local_estimate": round(new_total, 6),
        "prev_local": prev_total,
        "delta": round(new_total - prev_total, 6),
        "n_tasks_updated": len(updates),
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

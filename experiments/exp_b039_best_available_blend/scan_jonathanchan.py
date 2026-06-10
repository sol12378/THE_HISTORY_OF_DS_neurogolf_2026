"""
jonathanchan_v3 の静的コスト改善候補をすべてチェックし、
genuine な official ORT cost 改善を見つける。
セグフォルト回避のためプロセスを再起動しながら進む。
"""
import csv, json, math, pathlib, sys, subprocess, time, zipfile, tempfile, onnx
import numpy as np, onnxruntime as ort, importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng = importlib.util.module_from_spec(spec); spec.loader.exec_module(ng)

ARTIFACT_DIR = ROOT / "data" / "cache" / "public_artifacts"
BEST_MANIFEST = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "selected_manifest.csv"
BEST_ZIP = ROOT / "experiments" / "exp_b025_submit_safe_delta_union" / "submission.zip"

BANNED = {"LOOP","SCAN","NONZERO","UNIQUE","SCRIPT","FUNCTION","COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}
DT_BYTES = {1:4,2:1,3:1,4:2,5:4,6:8,7:8,8:16,9:1,10:4,11:8,12:4,13:4}

best_costs = {}
with open(BEST_MANIFEST) as f:
    for row in csv.DictReader(f):
        cr = row.get("cost",""); best_costs[int(row["task_id"])] = int(cr) if cr.strip() else 0

def static_cost(data):
    try:
        model = onnx.load_model_from_string(data)
        p = m = 0
        for t in model.graph.initializer:
            if not t.dims: continue
            n = math.prod(t.dims); p += n; m += n * DT_BYTES.get(t.data_type, 4)
        return m + p
    except: return None

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
        while row and row[-1] == 10: row.pop()
        result.append(row)
    while result and not result[-1]: result.pop()
    return result

def official_cost(model_bytes, ex_input):
    tmp_pfx = pathlib.Path(tempfile.mktemp(suffix="", prefix="ng_"))
    try:
        opts = ort.SessionOptions()
        opts.enable_profiling = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        opts.profile_file_prefix = str(tmp_pfx)
        opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
        sess.run(None, {sess.get_inputs()[0].name: grid_to_f32(ex_input)})
        trace = sess.end_profiling()
        model = onnx.load_model_from_string(model_bytes)
        mem, params = ng.score_network(model, trace)
        pathlib.Path(trace).unlink(missing_ok=True)
        return int(mem)+int(params) if mem and params else None
    except: return None
    finally:
        for p in tmp_pfx.parent.glob(tmp_pfx.name+"*.json"):
            try: p.unlink()
            except: pass

def validate(model_bytes, task):
    try:
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        sess = ort.InferenceSession(model_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        for split in ["train","test"]:
            for ex in task.get(split,[]):
                ig = ex["input"]
                if not ig or max(len(ig), max(len(r) for r in ig)) > 30: return False
                pred = sess.run(None, {inp: grid_to_f32(ig)})[0]
                if f32_to_grid(pred) != ex["output"]: return False
        return True
    except: return False

def normalize_io(model):
    real_inputs = [i for i in model.graph.input if not any(i.name==t.name for t in model.graph.initializer)]
    if not real_inputs or not model.graph.output: return model
    rename = {}
    if real_inputs[0].name != "input": rename[real_inputs[0].name]="input"; real_inputs[0].name="input"
    if model.graph.output[0].name != "output": rename[model.graph.output[0].name]="output"; model.graph.output[0].name="output"
    if rename:
        for node in model.graph.node:
            node.input[:] = [rename.get(x,x) for x in node.input]
            node.output[:] = [rename.get(x,x) for x in node.output]
        for v in model.graph.value_info: v.name = rename.get(v.name, v.name)
        for t in model.graph.initializer: t.name = rename.get(t.name, t.name)
    return model

pts = lambda c: max(1.0, 25.0 - math.log(max(1,c)))

# Check sources (ordered by potential)
SOURCES_TO_CHECK = [
    ("jonathanchan_v3", ARTIFACT_DIR / "jonathanchan_v3"),
    ("biohack44_6113_bundle", ARTIFACT_DIR / "biohack44_6113_bundle"),
    ("octaviograu_manual_v205", ARTIFACT_DIR / "octaviograu_manual_v205"),
    ("needless090_onnx_v31", ARTIFACT_DIR / "needless090_onnx_v31"),
    ("agentzz_6284_zip", ARTIFACT_DIR / "agentzz_6284_zip"),
    ("konbu17_v117", ARTIFACT_DIR / "konbu17_v117"),
    ("jvlegend_conservative_v136_plus13", ARTIFACT_DIR / "jvlegend_conservative_v136_plus13"),
]

improvements = []
t0 = time.time()

for src_name, src_base in SOURCES_TO_CHECK:
    src_dir = src_base / "submission" if (src_base / "submission").exists() else src_base
    if not src_dir.exists():
        continue
    print(f"\n=== Checking {src_name} ===", flush=True)

    for tid in range(1, 401):
        best_c = best_costs.get(tid, 0)
        if best_c == 0: continue  # perfect, skip

        fname = f"task{tid:03d}.onnx"
        p = src_dir / fname
        if not p.exists(): continue

        try:
            data = p.read_bytes()
        except: continue

        # Quick checks without full validation
        try:
            model = onnx.load_model_from_string(data)
            domains = {op.domain for op in model.opset_import}
            if not domains.issubset(ALLOWED_DOMAINS): continue
            ops = {n.op_type.upper() for n in model.graph.node}
            if ops & BANNED: continue
        except: continue

        sc = static_cost(data)
        if sc is None or sc == 0 or sc >= best_c: continue

        # Worth checking: static cheaper
        with open(DATA_DIR / f"task{tid:03d}.json") as f:
            task = json.load(f)

        try:
            model = normalize_io(onnx.load_model_from_string(data))
            model_bytes = model.SerializeToString()
        except: continue

        if not validate(model_bytes, task): continue

        oc = official_cost(model_bytes, task["train"][0]["input"])
        if oc and oc < best_c:
            gain = pts(oc) - pts(best_c)
            print(f"  IMPROVEMENT task{tid:03d}: {best_c}->{oc} +{gain:.4f}pts src={src_name} t={time.time()-t0:.0f}s", flush=True)
            improvements.append((tid, src_name, data, model_bytes, best_c, oc))

print(f"\n=== SCAN COMPLETE ===", flush=True)
print(f"Total improvements: {len(improvements)}", flush=True)
for tid, src, _, _, bc, oc in improvements:
    print(f"  task{tid:03d}: {bc}->{oc} +{pts(oc)-pts(bc):.4f}pts src={src}", flush=True)

# Save results
results_dir = pathlib.Path(__file__).parent
with open(results_dir / "scan_results.json", "w") as f:
    json.dump([{"task_id": t, "source": s, "best_c": bc, "new_c": oc, "gain": pts(oc)-pts(bc)}
               for t, s, _, _, bc, oc in improvements], f, indent=2)
print(f"Saved scan_results.json", flush=True)

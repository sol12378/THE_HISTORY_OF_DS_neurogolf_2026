"""Single-task probe: check if source has cheaper official ORT cost than best."""
import sys, json, math, pathlib, tempfile, onnx, onnxruntime as ort, numpy as np
import importlib.util

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "external" / "neurogolf-2026" / "unzipped"
UTILS_DIR = DATA_DIR / "neurogolf_utils"
sys.path.insert(0, str(UTILS_DIR))
spec = importlib.util.spec_from_file_location("neurogolf_utils", UTILS_DIR / "neurogolf_utils.py")
ng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ng)

BANNED = {"LOOP","SCAN","NONZERO","UNIQUE","SCRIPT","FUNCTION","COMPRESS"}
ALLOWED_DOMAINS = {"", "ai.onnx"}

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
        opts = ort.SessionOptions(); opts.log_severity_level=3
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

if __name__ == "__main__":
    # Args: tid src_path best_cost
    tid = int(sys.argv[1])
    src_path = pathlib.Path(sys.argv[2])
    best_cost = int(sys.argv[3])

    data = src_path.read_bytes()
    try:
        model = onnx.load_model_from_string(data)
        domains = {op.domain for op in model.opset_import}
        if not domains.issubset(ALLOWED_DOMAINS):
            print(f"SKIP:custom_domain:{domains}")
            sys.exit(0)
        ops = {n.op_type.upper() for n in model.graph.node}
        if ops & BANNED:
            print(f"SKIP:banned_op")
            sys.exit(0)
    except Exception as e:
        print(f"SKIP:parse:{e}")
        sys.exit(0)

    # Normalize IO
    real_inputs = [i for i in model.graph.input if not any(i.name==t.name for t in model.graph.initializer)]
    if not real_inputs or not model.graph.output:
        print("SKIP:no_io")
        sys.exit(0)
    rename = {}
    if real_inputs[0].name != "input": rename[real_inputs[0].name]="input"; real_inputs[0].name="input"
    if model.graph.output[0].name != "output": rename[model.graph.output[0].name]="output"; model.graph.output[0].name="output"
    if rename:
        for node in model.graph.node:
            node.input[:] = [rename.get(x,x) for x in node.input]
            node.output[:] = [rename.get(x,x) for x in node.output]
        for v in model.graph.value_info: v.name = rename.get(v.name, v.name)
        for t in model.graph.initializer: t.name = rename.get(t.name, t.name)
    model_bytes = model.SerializeToString()

    with open(DATA_DIR / f"task{tid:03d}.json") as f:
        task = json.load(f)

    if not validate(model_bytes, task):
        print("SKIP:validation_fail")
        sys.exit(0)

    oc = official_cost(model_bytes, task["train"][0]["input"])
    if oc is None:
        print("SKIP:cost_none")
        sys.exit(0)
    if oc >= best_cost:
        print(f"SKIP:not_cheaper:{oc}>={best_cost}")
        sys.exit(0)

    # SUCCESS
    print(f"OK:{oc}")
    # Write model bytes to temp file for parent to read
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".onnx")
    tmp.write(model_bytes)
    tmp.close()
    print(f"MODEL:{tmp.name}")

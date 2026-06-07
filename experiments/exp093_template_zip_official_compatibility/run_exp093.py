from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.phase1_rewrite_utils import infer_static_ok, load_neurogolf_utils, score_model, sha256  # noqa: E402


EXP_ID = "exp093_template_zip_official_compatibility"
EXP_DIR = ROOT / "experiments" / EXP_ID
ZIP_PATH = pathlib.Path(r"C:\Users\doran\Downloads\neurogolf_templates.zip")
UNPACK_DIR = EXP_DIR / "neurogolf_templates"
TASK_ID_FOR_SCORE = 48


@dataclass(frozen=True)
class TemplateRow:
    name: str
    status: str
    input_name: str
    output_name: str
    input_type: str
    output_type: str
    input_shape: str
    output_shape: str
    node_count: int
    initializer_count: int
    official_static_ok: bool
    official_static_reason: str
    official_score_memory: int | str
    official_score_params: int | str
    official_score_cost: int | str
    official_score_reason: str
    official_runtime_reason: str
    local_uint8_runtime_reason: str
    file_bytes: int
    sha256: str


def clean_extract() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if UNPACK_DIR.exists():
        shutil.rmtree(UNPACK_DIR)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if not member.startswith("neurogolf_templates/"):
                continue
            rel = pathlib.Path(member).relative_to("neurogolf_templates")
            target = UNPACK_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))


def import_module(path: pathlib.Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def type_name(elem_type: int) -> str:
    return onnx.TensorProto.DataType.Name(elem_type)


def shape_str(value_info: Any) -> str:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(str(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return "x".join(dims)


def official_runtime(raw: bytes) -> str:
    try:
        sess = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
        x = np.zeros((1, 10, 30, 30), dtype=np.float32)
        sess.run(None, {"input": x})
        return "ok"
    except Exception as exc:
        return f"failed: {str(exc)[:220]}"


def local_runtime(model: onnx.ModelProto) -> str:
    try:
        inp = model.graph.input[0]
        elem = inp.type.tensor_type.elem_type
        dtype = np.uint8 if elem == onnx.TensorProto.UINT8 else np.int32 if elem == onnx.TensorProto.INT32 else np.float32
        x = np.arange(24, dtype=dtype).reshape(4, 6) % 10
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        y = sess.run(None, {inp.name: x})[0]
        return f"ok shape={tuple(y.shape)} dtype={y.dtype}"
    except Exception as exc:
        return f"failed: {str(exc)[:220]}"


def evaluate_model(utils: Any, name: str, model: onnx.ModelProto) -> TemplateRow:
    raw = model.SerializeToString()
    (EXP_DIR / f"{name}.onnx").write_bytes(raw)
    inp = model.graph.input[0]
    out = model.graph.output[0]
    static_ok, static_reason = infer_static_ok(model)
    memory: int | None = None
    params: int | None = None
    score_reason = ""
    if static_ok:
        memory, params, score_reason = score_model(utils, raw, TASK_ID_FOR_SCORE, name, EXP_DIR)
    else:
        score_reason = "not scored: official static check failed"
    cost: int | str = "" if memory is None or params is None else int(memory) + int(params)
    return TemplateRow(
        name=name,
        status="official_compatible" if static_ok and memory is not None and official_runtime(raw) == "ok" else "not_official_compatible",
        input_name=inp.name,
        output_name=out.name,
        input_type=type_name(inp.type.tensor_type.elem_type),
        output_type=type_name(out.type.tensor_type.elem_type),
        input_shape=shape_str(inp),
        output_shape=shape_str(out),
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        official_static_ok=static_ok,
        official_static_reason=static_reason,
        official_score_memory="" if memory is None else int(memory),
        official_score_params="" if params is None else int(params),
        official_score_cost=cost,
        official_score_reason=score_reason,
        official_runtime_reason=official_runtime(raw),
        local_uint8_runtime_reason=local_runtime(model),
        file_bytes=len(raw),
        sha256=sha256(raw),
    )


def main() -> None:
    clean_extract()
    utils = load_neurogolf_utils()
    sys.path.insert(0, str(UNPACK_DIR))
    T = import_module(UNPACK_DIR / "templates.py", "zip_templates_basic")
    L = import_module(UNPACK_DIR / "templates_logical.py", "zip_templates_logical")
    R = import_module(UNPACK_DIR / "templates_recolor_dynamic.py", "zip_templates_recolor")
    models = {
        "identity": T.t_identity(),
        "flip_lr": T.t_flip_lr(),
        "rot180": T.t_rot180(),
        "crop": T.t_crop(1, 3, 1, 4),
        "recolor": T.t_recolor([0, 1, 3, 2, 4, 5, 6, 7, 8, 9]),
        "logical_recolor_where": L.l_recolor_where(2, 8),
    }
    if hasattr(R, "recolor_direct"):
        models["recolor_direct"] = R.recolor_direct([0, 1, 3, 2, 4, 5, 6, 7, 8, 9])
    rows = [evaluate_model(utils, name, model) for name, model in models.items()]
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "compatibility_audit_ready",
        "zip_path": str(ZIP_PATH),
        "template_count_tested": len(rows),
        "official_compatible_count": sum(1 for row in rows if row.status == "official_compatible"),
        "rows": [asdict(row) for row in rows],
        "decision": (
            "The zip templates are useful as design inspiration for uint8/minimal-tensor thinking, "
            "but not directly usable for current NeuroGolf submissions because official validation expects input/output names "
            "`input`/`output`, FLOAT one-hot [1,10,30,30] tensors, and static shapes. "
            "Adapting them requires wrapper conversion from one-hot to integer grid and back, which likely destroys the advertised 250-600 costs."
        ),
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: compatibility audit only",
        "leakage_risk": "low: external templates inspected; no task labels encoded.",
        "overfitting_risk": "low: no candidate submission emitted.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = f"""# {EXP_ID}

## 目的

`C:\\Users\\doran\\Downloads\\neurogolf_templates.zip` のテンプレートが、現在の公式NeuroGolf utilsでそのまま `cost<=250〜600` の提出候補になり得るかを調査する。

## 結果

- tested templates: `{len(rows)}`
- official compatible: `{result["official_compatible_count"]}`

## 解釈

zip内テンプレートは `uint8 [H,W]` または `int32 [H,W]` の整数グリッドを想定している。一方、現在の公式 `convert_to_numpy` / `verify_subset` は `FLOAT [1,10,30,30]` one-hot tensorを直接比較する。入出力名もzip側は `in`/`out`、公式側は `input`/`output` を前提にしている。

したがって、zipテンプレートの低cost値はこのworkspaceの公式評価契約とは直接一致しない。使うなら、個別taskでone-hot公式表現へ移植し、公式 `score_network` で再計測する必要がある。

## Decision

直接提出候補としては不採用。設計思想、特に「中間テンソル数を減らす」「uint8/整数グリッドなら安い」という方向性は参考にする。ただしwrapper変換を足すと250〜600の利点は崩れる可能性が高い。

## Risk

- leakage risk: low。
- overfitting risk: low。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

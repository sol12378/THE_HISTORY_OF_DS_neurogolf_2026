from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_ID = "exp_b023_seddik_surgery_pattern_audit"
EXP_DIR = ROOT / "experiments" / EXP_ID
NB_PATH = (
    ROOT
    / "experiments"
    / "exp_b022_public_notebook_intelligence"
    / "notebooks"
    / "seddiktrk_surgical_precision"
    / "surgical-onnx-precision-parameter-reduction.ipynb"
)


@dataclass(frozen=True)
class PatternRow:
    pattern: str
    evidence_count: int
    useful_interpretation: str
    exp_b_action: str
    risk: str


def notebook_text() -> tuple[str, list[str]]:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code":
            cells.append(src)
    return "\n\n".join(cells), cells


def context_snippets(text: str, pattern: str, radius: int = 240) -> list[str]:
    snippets = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        s = max(0, m.start() - radius)
        e = min(len(text), m.end() + radius)
        snippets.append(text[s:e].replace("\n", "\\n"))
    return snippets[:10]


def main() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    text, cells = notebook_text()
    patterns = [
        (
            "initializer_pruning",
            r"initializer|unused|graph\.initializer",
            "未使用initializerやinitializer型/shapeの削減は、exp023/b021系の安全なgraph surgeryに近い。",
            "strict seed全taskでunused initializer / duplicate initializer / scalar initializer化を再監査する。",
            "low-to-medium: validationとstatic check必須。",
        ),
        (
            "compress_or_banned_op_awareness",
            r"Compress|NonZero|Unique|Sequence",
            "公開notebookにはbanned/危険opへの言及があり、strict filterのguardrailとして使える。",
            "candidate emission前にbanned op検出を強化し、Compress系artifactをsubmit candidateから外す。",
            "low: guardrail強化。",
        ),
        (
            "precision_parameter_reduction",
            r"precision|float16|float32|float64|parameter|params",
            "parameter reductionやdtype削減はcostに効く可能性があるが、公式入力/出力float one-hotとの整合が必要。",
            "strict seedからinitializer dtype/constant size削減候補を生成し、full-arc gateで評価する。",
            "medium: dtype変更は数値/argmax挙動を壊す可能性。",
        ),
        (
            "score_cost_loop",
            r"score|cost|profile|onnxruntime|InferenceSession",
            "score loopの構造はofficial-like profilingの再現に役立つ。",
            "exp_bのcandidate evaluationにprofile/cost failure reasonをより細かく記録する。",
            "low.",
        ),
        (
            "pad_conv_where_patterns",
            r"Pad|Conv|Where|Slice|Gather",
            "Pad/Conv/Where/Slice/Gatherの使用頻度から、cheap loweringとheavy loweringの境界を読む。",
            "massimiliano/vyanktesh notebookと合わせてcheap Slice/Gather/Pad lowering patternを抽出する。",
            "medium: naive full-grid patternsはb011のように高cost化。",
        ),
    ]
    rows = []
    snippets = {}
    for name, regex, interp, action, risk in patterns:
        matches = re.findall(regex, text, flags=re.IGNORECASE)
        rows.append(PatternRow(name, len(matches), interp, action, risk))
        snippets[name] = context_snippets(text, regex)
    function_names = re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    import_counts = Counter(re.findall(r"^\s*(?:import|from)\s+([A-Za-z0-9_\.]+)", text, flags=re.MULTILINE))
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "pattern_audit_ready",
        "notebook": str(NB_PATH.relative_to(ROOT)),
        "code_cell_count": len(cells),
        "function_names": function_names,
        "import_counts": dict(import_counts),
        "patterns": [asdict(r) for r in rows],
        "snippets": snippets,
        "decision": "Prioritize initializer/parameter surgery audit on strict seed, then dtype/constant-size reductions only with full-arc validation.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: intelligence audit only",
        "leakage_risk": "medium: public notebook intelligence; use only as pattern hints.",
        "overfitting_risk": "low if used as guardrails; medium if copied as artifacts.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    import csv

    with (EXP_DIR / "pattern_rows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PatternRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([asdict(r) for r in rows])
    notes = "# exp_b023_seddik_surgery_pattern_audit\n\n"
    notes += "## 目的\n\n`seddiktrk/surgical-onnx-precision-parameter-reduction` を、strict seedへ使えるgraph surgery/precision削減patternの知識源として読む。\n\n"
    notes += "## 抽出pattern\n\n"
    for row in rows:
        notes += f"- `{row.pattern}`: count={row.evidence_count}; action={row.exp_b_action}; risk={row.risk}\n"
    notes += "\n## Decision\n\nまずstrict seed全taskに対して、unused/duplicate/scalar initializerなど低リスクなparameter surgery auditを行う。dtype変更はfull-arc gate付きの後段に回す。\n"
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

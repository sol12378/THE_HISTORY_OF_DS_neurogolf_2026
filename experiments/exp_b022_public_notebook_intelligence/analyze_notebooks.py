from __future__ import annotations

import csv
import json
import pathlib
import re
from collections import Counter
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[2]
EXP_ID = "exp_b022_public_notebook_intelligence"
EXP_DIR = ROOT / "experiments" / EXP_ID
NOTEBOOK_DIR = EXP_DIR / "notebooks"


KEYWORDS = [
    "Conv",
    "Slice",
    "Gather",
    "Scatter",
    "MatMul",
    "precision",
    "float16",
    "initializer",
    "blend",
    "submission",
    "onnx",
    "rule",
    "description",
    "cost",
    "score",
    "Compress",
    "Where",
    "Pad",
    "Resize",
    "surgery",
    "parameter",
]


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def notebook_name(path: pathlib.Path) -> str:
    rel = path.relative_to(NOTEBOOK_DIR)
    return rel.parts[0]


def classify(name: str, text: str) -> tuple[str, str, str]:
    explicit = {
        "massimilianoghiotto_convolution_series_part_3": (
            "conv_lowering_patterns",
            "Convolution seriesからsmall Conv/Pad/Slice loweringのbest practiceを抽出する",
            "small Conv kernels / padding discipline / local predicate lowering",
        ),
        "konbu17_blended_till_4_27": (
            "blend_source_inventory",
            "blend構成とsource採用順をstrict seed / LB calibration観点で読む",
            "source reliability / task-level delta / duplicate-source audit",
        ),
        "vyanktesh_multi_source": (
            "multi_source_solver_patterns",
            "multi-source ONNX solverのtask選択・Slice/Gather/Pad実装を読む",
            "source reliability plus cheap Slice/Gather/Pad lowering patterns",
        ),
        "seddiktrk_surgical_precision": (
            "onnx_surgery_precision",
            "surgical precision/parameter reductionから安全なgraph surgery候補を抽出する",
            "initializer pruning / dtype or parameter reduction / static safety guardrails",
        ),
        "karnak_all_task_description": (
            "task_description_taxonomy",
            "all-task description analysisをrule family/lane分類に使う",
            "family taxonomy / task description prompts / rule-search target queue",
        ),
        "magmacot_new_blending": (
            "blend_source_inventory",
            "new blendingのsource構成をstrict seedと比較して信頼できるdelta候補を探す",
            "blend source audit / micro-delta candidate selection",
        ),
        "needless090_4250": (
            "baseline_source_inventory",
            "低スコアだが単純・低riskなsourceとして、strict seed差分や初期taskを確認する",
            "low-risk source audit / baseline duplicate check",
        ),
    }
    if name in explicit:
        return explicit[name]
    low = (name + "\n" + text[:20000]).lower()
    if "surgical" in low or "precision" in low or "parameter" in low:
        return (
            "onnx_surgery_precision",
            "既存ONNXのprecision/parameter削減やsurgery候補を読む",
            "graph surgery / initializer pruning / dtype reduction guardrail",
        )
    if "convolution" in low or "conv" in low:
        return (
            "conv_lowering_patterns",
            "Conv seriesの構造からsmall Conv loweringのbest practiceを抽出する",
            "small Conv kernels / color map / local predicate lowering",
        )
    if "blend" in low or "multi-source" in low or "source" in low:
        return (
            "blend_source_inventory",
            "public source blendの採用順・重複・riskを整理する",
            "strict seed calibration / source reliability / task-level deltas",
        )
    if "description" in low or "analysis" in low:
        return (
            "task_description_taxonomy",
            "task descriptionからrule family/lane分類を改善する",
            "family taxonomy / rule-search prompts / target queue",
        )
    return ("general_public_reference", "補助情報として読む", "manual review")


def main() -> None:
    files = [p for p in NOTEBOOK_DIR.rglob("*") if p.is_file()]
    file_rows = []
    notebook_text: dict[str, str] = {}
    for path in files:
        text = read_text(path) if path.suffix.lower() in {".py", ".ipynb", ".json", ".md", ".txt"} else ""
        nb = notebook_name(path)
        notebook_text[nb] = notebook_text.get(nb, "") + "\n" + text[:50000]
        counts = Counter()
        for kw in KEYWORDS:
            counts[kw] = len(re.findall(re.escape(kw), text, flags=re.IGNORECASE))
        file_rows.append(
            {
                "notebook": nb,
                "path": str(path.relative_to(ROOT)),
                "suffix": path.suffix,
                "bytes": path.stat().st_size,
                "keyword_counts": json.dumps({k: v for k, v in counts.items() if v}, ensure_ascii=False),
            }
        )
    notebook_rows = []
    for nb, text in sorted(notebook_text.items()):
        counts = Counter()
        for kw in KEYWORDS:
            counts[kw] = len(re.findall(re.escape(kw), text, flags=re.IGNORECASE))
        category, useful_use, action = classify(nb, text)
        notebook_rows.append(
            {
                "notebook": nb,
                "category": category,
                "useful_use": useful_use,
                "next_action": action,
                "total_text_chars": len(text),
                "keyword_counts": json.dumps({k: v for k, v in counts.items() if v}, ensure_ascii=False),
            }
        )
    result = {
        "exp_id": EXP_ID,
        "date": date.today().isoformat(),
        "status": "intelligence_ready",
        "downloaded_notebook_count": len(notebook_rows),
        "downloaded_file_count": len(file_rows),
        "notebooks": notebook_rows,
        "decision": "Use these notebooks as structured intelligence: blend notebooks for source reliability, Conv/surgery notebooks for lowering, description analysis for rule taxonomy.",
        "local_estimate_delta": 0.0,
        "submission_decision": "no_submit: intelligence gathering only",
        "leakage_risk": "medium: public notebooks may contain leaderboard-tuned artifacts; use as hints and validate with strict/full-arc/LB delta.",
        "overfitting_risk": "medium-high if copied blindly; low if used only for rule/lowering ideas with independent validation.",
    }
    (EXP_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (EXP_DIR / "downloaded_files.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["notebook", "path", "suffix", "bytes", "keyword_counts"])
        writer.writeheader()
        writer.writerows(file_rows)
    with (EXP_DIR / "notebook_intelligence.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["notebook", "category", "useful_use", "next_action", "total_text_chars", "keyword_counts"])
        writer.writeheader()
        writer.writerows(notebook_rows)
    notes = "# exp_b022_public_notebook_intelligence\n\n"
    notes += "## 目的\n\nユーザー指定のKaggle notebookを取得し、blend/sourceとしてだけでなくrule/lowering知識として使えるように分類する。\n\n"
    notes += "## Notebook Intelligence\n\n"
    for row in notebook_rows:
        notes += f"- `{row['notebook']}`: {row['category']} / {row['useful_use']} / next: {row['next_action']}\n"
    notes += "\n## Decision\n\n公開notebookはそのままcopyするのではなく、strict/full-arc validationと小delta LB calibrationを通した知識源として使う。\n"
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

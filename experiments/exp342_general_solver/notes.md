# exp342 — P3-1 Layer-A 汎用ソルバ + LLM proposer + Layer-D 被覆計測

/ 目的: 計画書 §4層A / §8 P3-1。enumerative DSL ソルバ + LLM(Qwen) proposer を構築し、
未解決タスクの**規則化率を計測可能**にする(受け入れ基準)。検証は決定的、採択は層E gate。

## 実装
- **`neurogolf_solver.py`**: param駆動 DSL(geometric/recolor/crop/tile)。
  - numpy で apply/verify(全 arc-gen で決定的検証)。
  - **generic chained lowerer**: fixed-input-shape タスクを (1,10,30,30) 上で合成ONNX化
    (Slice→atoms→Pad)。可変shapeは static gate(dynamic不可)で lower 不可。
  - enumerate_programs: geometric(×recolor) depth≤2 を全例verifyで探索。
- **`neurogolf_llm_proposer.py`**: 既存 `autoresearch.llm.LLMClient`(vLLM @localhost:8000,
  **Qwen3.5-35B-A3B-GPTQ-Int4 稼働確認**)。train例を grid テキスト化→DSL仕様提示→
  JSON program を生成→ローカルでパース。**LLMは proposer のみ、検証は決定的**(verify_program)。
  endpoint 不通時は [] を返し enumerative core にdegrade。

## 計測結果(Layer-D 完全性クリティック)
**enumerative core(LLMなし, 全393非frozen):**
- **規則化 11/393 = 2.8%** / lowered+valid 5 / **WINS(union超え)=0**。
- stuck内訳: **no_rule 382(96.7%)** / rule_not_lowerable 6(可変shape) / lowered_not_cheaper 5。

**LLM proposer(Qwen, end-to-end実証):**
- **task087: Qwen が rot180 を提案→verify成功(28.7s)** = LLMが正しい規則を生成できることを実証。
- task116: 提案パース0(規則がDSL外, 92s)。**1コール ~30-90s と遅く、DSL表現力に上限**。

## 結論
ソルバ枠組み(enumerative + 実LLM proposer + 決定的検証 + gated lowering + 規則化計測 +
完全性クリティック)は**完成・end-to-end動作**。だが:
1. **DSLが狭い**(geometric/recolor/crop/tileのみ)→ 96.7%が規則化不能(ARC本来の難しさ)。
2. **規則化できても union を超えない**(0win)= P1/P2と同じ「unionは十分golf済」。
3. LLMは正しく規則提案できるが**遅く(~60s/call)・同じDSL上限**に縛られる。
→ #1到達には **遥かに豊かなDSL + 各規則の安価lowering** が必要で、これは ARC Prize 級の大型研究。
本枠組みはその土台(系統的探索・LLM統合・計測)を提供する。

## 統合方針(安全)
- ソルバは**自律sweepに未配線**(0win + LLM ~60s/call で sweep が極端に重くなる)。emitter同様、
  win が出る見込みが立った時点で配線。`run_solver_coverage.py --llm N` で運用的に LLM 被覆拡大可能。
- 全候補は `evaluate_candidate`(static+full-arc(-1)+cost厳密減)経由 = 安全。提出なし・union/baseline無変更。

## 再現
`test_solver_core.py`(enum+lower+gate) / `test_llm_proposer.py`(Qwen end-to-end) /
`run_solver_coverage.py [--llm N]`(規則化率+完全性クリティック)。
PYTHONPATH に `.venv/Lib/site-packages;autoresearch/src`(LLMClient用)。

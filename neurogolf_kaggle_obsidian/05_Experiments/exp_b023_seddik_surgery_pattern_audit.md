# exp_b023_seddik_surgery_pattern_audit

## 目的

`seddiktrk/surgical-onnx-precision-parameter-reduction` を、strict seedへ使えるgraph surgery / precision削減patternの知識源として読む。

## 抽出結果

- code cells: `20`
- functions:
  - `score_task`
  - `cleanup_unused_initializers`
  - `cleanup_all_tasks`
  - `initializer_key`
  - `find_duplicate_initializers`
  - `deduplicate_initializers`
  - `find_compressible_initializers`
  - `compress_uniform_initializers`

## 有用pattern

- `initializer_pruning`
  - evidence count: `53`
  - action: strict seed全taskでunused / duplicate / scalar initializer化を再監査する。
- `compress_or_banned_op_awareness`
  - evidence count: `15`
  - action: Compressやbanned opをsubmit候補から外すguardrailを強化する。
- `precision_parameter_reduction`
  - evidence count: `56`
  - action: initializer dtype / constant size削減候補をfull-arc gateで評価する。
- `score_cost_loop`
  - evidence count: `45`
  - action: exp_b candidate evaluationにprofile/cost failure reasonを細かく記録する。
- `pad_conv_where_patterns`
  - action: massimiliano/vyanktesh notebookと合わせてcheap Slice/Gather/Pad/Conv loweringを読む。

## Decision

次は `safe uniform initializer scalarization` をstrict seed全taskへ試す。seddik notebookにもskip例があるため、blind適用せず、candidateごとにstatic check + full arc-gen validation + official-like scoreを通す。

## Risk

- leakage risk: medium。公開notebook由来の知識。
- overfitting risk: low if guardrail only; medium if copied as artifacts。

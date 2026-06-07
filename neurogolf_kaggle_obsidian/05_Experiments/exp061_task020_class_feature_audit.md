# exp061_task020_class_feature_audit

## Hypothesis

task020の3-template classは入力target colorの位置特徴から分離できる。

## Result

- examples: `266`
- template counts: `corners 87`, `inner_diag 87`, `edge_mid 92`
- best keys:
  - `pos`: 71 keys, accuracy `1.000`
  - `canon_pos`: 24 keys, accuracy `1.000`
  - `canon_and_count`: 24 keys, accuracy `1.000`
  - `row_col_counts`: 71 keys, accuracy `1.000`
- lower-cardinality heuristic:
  - `touch_signature`: 8 keys, accuracy `0.797`
  - `count_and_touch`: 14 keys, accuracy `0.820`

## Interpretation

`canon_pos` で完全分離できるが、24-key tableをそのまま提出するとlookup/memorization riskが高い。次は24 keyを説明可能な幾何ruleへ圧縮する。

## Decision

raw lookupとしてONNX化しない。`canon_pos` の24 caseを、corner/edge/inner orbitの包含・欠損・接触条件へ圧縮する。

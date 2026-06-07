# exp100_low_cost_artifact_rule_mining

## 目的

既存のcost<=2000 artifactをNeuroGolf専用compilerのgrammarへ変換する。単なるprofileではなく、archetypeとrewrite directionを機械可読にする。

## 結果

- selected artifacts: `83`
- rewrite rule candidates: `11`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Top Archetypes

[
  [
    "one_node_conv_kernel",
    20
  ],
  [
    "computed_slice_pad",
    20
  ],
  [
    "misc_under_2000",
    11
  ],
  [
    "one_node_gridsample",
    9
  ],
  [
    "one_node_gather_index_map",
    8
  ],
  [
    "channel_gather_colormap",
    4
  ],
  [
    "static_slice_pad",
    4
  ],
  [
    "one_node_transpose",
    2
  ],
  [
    "misc_under_600",
    2
  ],
  [
    "tiny_dynamic_shape_index",
    2
  ]
]

## Decision

次は `static_slice_pad`, `channel_gather_colormap`, `one_node_conv_kernel`, `tiny_dynamic_shape_index` を候補生成器として実装し、exp087の小出力候補へ流す。

## Risk

- leakage risk: low。
- overfitting risk: catalog段階ではlow。候補生成後はfull validation必須。

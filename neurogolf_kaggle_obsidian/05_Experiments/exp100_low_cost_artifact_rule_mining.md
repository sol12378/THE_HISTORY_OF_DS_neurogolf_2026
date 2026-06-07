# exp100_low_cost_artifact_rule_mining

## 目的

既存のcost<=2000 artifactをNeuroGolf専用compilerのgrammarへ変換する。単なるprofileではなく、archetypeとrewrite directionを機械可読にする。

## 結果

- selected artifacts: `83`
- rewrite rule candidates: `11`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Top Archetypes

- `one_node_conv_kernel`: 20
- `computed_slice_pad`: 20
- `misc_under_2000`: 11
- `one_node_gridsample`: 9
- `one_node_gather_index_map`: 8
- `channel_gather_colormap`: 4
- `static_slice_pad`: 4

## Decision

最初のcompiler grammarは `static_slice_pad`, `channel_gather_colormap`, `one_node_conv_kernel`, `computed_slice_pad`, `tiny_dynamic_shape_index` に絞る。汎用op探索ではなく、低cost artifactから逆輸入した文法を使う。

## Risk

- leakage risk: low。
- overfitting risk: catalog段階ではlow。候補生成後はfull validation必須。

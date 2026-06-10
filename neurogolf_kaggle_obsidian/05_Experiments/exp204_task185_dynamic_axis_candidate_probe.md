# exp204_task185_dynamic_axis_candidate_probe

## 目的

task185でdilated axis selectorから`GatherElements`用index tensorを作り、4x4 lattice extraction + homogeneous coreまで接続したcorrectness-first candidateを検証する。

## 結果

- `dynamic_axis_basic`: validation `0_pass_1_fail`, cost `106720`
- `dynamic_axis_nonzero_only`: validation `0_pass_1_fail`, cost `142730`
- `dynamic_axis_score_nonzero_core_basic`: validation `0_pass_1_fail`, cost `142730`
- `dynamic_axis_score_nonzero_default_bg`: validation `0_pass_1_fail`, cost `143490`
- accepted_count: `0`

## 判断

初回dynamic candidateは不採用。巨大index templateによりcostがbaseline `59584` を超え、correctnessもexample 0で失敗。次はcompact start/spacing tableによるindex導出と、selector/core位置対応のdebugが必要。

## リスク

- leakage risk: low。入力geometryのみ。
- overfitting risk: medium。task-specific geometryとdebug中の背景処理に依存。

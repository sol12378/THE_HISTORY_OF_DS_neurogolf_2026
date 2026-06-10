# exp314_grid_sample_priority_targets

## 目的

Phase C の `grid_sample` lane を実 task へ接続するため、farm smoke result に優先 target queue と cost 600 化時の期待 delta を出す。

## 結果

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- priority targets:
  - task251: current_cost `100580`, expected_delta `5.121876`
  - task037: current_cost `63726`, expected_delta `4.665371`
  - task185: current_cost `59584`, expected_delta `4.598086`
  - task048: current_cost `5481`, expected_delta `2.212843`
- submission: なし

## 判断

score-producing 変更ではないが、次の `grid_sample` supplier 実装の対象を明示できた。優先順位は task251 > task037 > task185 > task048。

## リスク

- leakage risk: low
- overfitting risk: low

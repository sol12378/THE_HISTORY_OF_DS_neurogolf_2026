# exp200_task185_axis_separable_selector_audit

## 目的

task185のpairwise window scoringが高costだったため、row/col windowを独立にscoreして同じselectorを再現できるか確認する。

## 結果

- pairwise_reference_pass: `267/267`
- best_axis_mode: `score`
- best_axis_pass: `267/267`
- mode_pass: `{'score': 267, 'score_active': 267, 'active_score': 267, 'rightmost_score': 267}`
- row_target_rank_hist: `{0: 267}`
- col_target_rank_hist: `{0: 267}`

## 判断

task185 selectorはpairwise row-window x col-window scoreを必要としない。row/colを独立にspecial-cell score最大で選べるため、ONNX loweringではpairwise `GatherND/ArgMax` を避ける。

## リスク

- leakage risk: low。入力geometryのみ。
- overfitting risk: medium-low。raw coordinate tableではないが、tie-breakの分布依存は残る。

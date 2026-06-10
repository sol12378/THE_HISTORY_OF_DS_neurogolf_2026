# exp200_task185_axis_separable_selector_audit

## 目的

task185のpairwise window scoringが高costだったため、row/col windowを独立にscoreして同じselectorを再現できるか確認する。

## 結果

- pairwise_reference_pass: `267/267`
- best_axis_mode: `score`
- best_axis_pass: `267/267`
- mode_pass: `{'score': 267, 'score_active': 267, 'active_score': 267, 'rightmost_score': 267}`

## 判断

full passならaxis-separable selectorとしてONNX化へ進む。未達ならpairwise相互作用が必要で、task185は別の低cost selector案か他taskへpivotする。

## リスク

- leakage risk: low。入力geometryだけを使う監査。
- overfitting risk: medium-low。raw coordinate tableではないが、selector tie-breakはarc-gen分布に依存しうる。

# exp247_task100_bbox_area_color_cost_probe

## 目的

exp246で発見したtask100 rule「bbox_area最大色を2x2全同色で出力」をONNX化し、baseline cost `6536` より安くなるか確認する。

## 結果

- validation: `266_pass_0_fail`
- candidate_cost: `74548`
- baseline_cost: `6536`
- status: `no_cost_gain`

## 判断

ruleは正しいが、row/col spanによるbbox_area selectorが重すぎる。task100はsolved-rule assetとして保持し、現loweringでは提出しない。

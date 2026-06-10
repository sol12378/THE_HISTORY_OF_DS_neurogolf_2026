# exp247_task100_bbox_area_color_cost_probe

## 目的

task100 rule「bbox_area最大の非zero色を選び、2x2全同色templateを出力」をONNX化し、baseline cost `6536` より安いか確認する。

## 結果

- validation: `266_pass_0_fail`
- status: `no_cost_gain`
- candidate_cost: `74548`
- baseline_cost: `6536`
- reason: `candidate cost is not lower than baseline`

## 判断

improvedならsingle-task delta提出候補。no_cost_gain/validation failならtask100はsolved assetとして保持。

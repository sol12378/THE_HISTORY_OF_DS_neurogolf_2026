# exp218_crop3x3_wide_rule_sweep

## 目的

3x3固定cropish候補に、安いSlice/transformでloweringできるcrop ruleを横展開する。

## 結果

- evaluated: 24 task
- full_hits:
  - task039 `bbox_top_left 3x3` `264/264`
  - task135 `top_right 3x3` `266/266`
- near_hits: none

## 判断

task039はbaseline cost `7772` で改善余地があるためcost probeへ進める。task135はbaseline cost `360` で優先度低。

## リスク

- leakage risk: low
- overfitting risk: medium-low

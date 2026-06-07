# exp117_task185_window_scoring_cost_probe

## Hypothesis

task185 window selectorをONNX化する前に、line detection、window scoring、spacing branch coreのcost floorを測る。

## Result

- campaign index: `19`
- `line_mask_conv_proxy`: cost `4204`
- `window_score_static_positions_proxy`: cost `76194`
- `three_branch_core_stack_proxy`: cost `5160`
- local delta: `0.000000`

## Interpretation

exp116のPython selectorは `267/267` で正しいが、素直なONNX loweringは重い。特に `GatherND/ArgMax` で全window pairをscoreするrouteは、現行task185 cost `59584` より悪化する。

line detectionやbranchless spacing coreも600級から遠い。task185を続けるなら、よりfusedなselector、shape/spacing特化、または既存artifactの別構造利用が必要。#20 reviewで、このlaneを継続するかtask366/task365へpivotするか判断する。

## Risk

- leakage risk: low。input-only cost proxies。
- overfitting risk: medium。static-position proxyは診断用で提出候補ではない。

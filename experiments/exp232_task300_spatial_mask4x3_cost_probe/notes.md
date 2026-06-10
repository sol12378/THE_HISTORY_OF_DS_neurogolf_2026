# exp232_task300_spatial_mask4x3_cost_probe

## 目的

task300 max-color proxyのGatherElements対象を10chから1ch spatial maskへ縮小し、exp231のcost `92595` をbaseline `77546` 未満に落とせるか確認する。

## 結果

- validation: `267_pass_0_fail`
- status: `no_cost_gain`
- candidate_cost: `81541`
- baseline_cost: `77546`
- reason: `candidate cost is not lower than baseline`

## 判断

improvedならbundle/submission候補。no_cost_gainなら更なるdtype/graph optimizationが必要。

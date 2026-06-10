# exp226_task300_max_color_mask4x3_cost_probe

## 目的

task300のmax-color proxy ruleをONNX化し、baseline cost `77546` より安くなるか確認する。

## 結果

- validation: `0_pass_1_fail`
- status: `rejected`
- candidate_cost: ``
- baseline_cost: `77546`
- reason: `mismatch example 0`

## 判断

improvedならbundle/submission候補。no_cost_gainなら正しいruleでもこのloweringはcost wall。

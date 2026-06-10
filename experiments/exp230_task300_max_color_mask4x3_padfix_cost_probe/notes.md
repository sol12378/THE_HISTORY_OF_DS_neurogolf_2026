# exp230_task300_max_color_mask4x3_padfix_cost_probe

## 目的

exp228のpadding外側channel0過剰を修正し、task300 max-color proxy ONNX候補のvalidation/costを再評価する。

## 結果

- validation: `1_pass_1_fail`
- status: `rejected`
- candidate_cost: ``
- baseline_cost: `77546`
- reason: `mismatch example 1`

## 判断

improvedならbundle/submission候補。no_cost_gainなら正しいruleでもこのloweringはcost wall。

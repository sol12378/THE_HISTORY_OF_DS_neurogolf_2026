# exp228_task300_max_color_mask4x3_bgfix_cost_probe

## 目的

exp226の背景channel欠落を修正し、task300 max-color proxy ONNX候補のvalidation/costを再評価する。

## 結果

- validation: `0_pass_1_fail`
- status: `rejected`
- candidate_cost: ``
- baseline_cost: `77546`
- reason: `mismatch example 0`

## 判断

improvedならbundle/submission候補。no_cost_gainなら正しいruleでもこのloweringはcost wall。

# exp239_task271_component_output_audit

## 目的

exp223で3x3固定かつone_component_shape `266/267` だったtask271について、component crop/maskまたはmax-color/largest proxyで内容まで説明できるか監査する。

## 結果

- baseline_cost: `28991`
- output_shape: `3x3` 固定
- output_color_count_hist: `{2: 267}`
- binary_signature_count: `1`
- any_same_shape_input_crop_exact: `267/267`
- component_crop_exact: `218/267`
- largest_crop_top_left: `106/267`
- max_color_crop_top_left: `28/267`

## 判断

task271はcomponent cropだけでは不十分だが、出力は全例で入力内の3x3 exact crop。次はcrop offset selectorを監査する。

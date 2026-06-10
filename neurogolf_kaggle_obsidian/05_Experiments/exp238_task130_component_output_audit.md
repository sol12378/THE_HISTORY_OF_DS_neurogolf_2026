# exp238_task130_component_output_audit

## 目的

exp223でshape-full候補だったtask130について、task300と同じcomponent crop/maskまたはmax-color proxyで内容まで説明できるか確認する。

## 結果

- baseline_cost: `15948`
- example_count: `265`
- output_shape: `3x3` 固定
- output_color_count_hist: `{1: 37, 2: 44, 3: 35, 4: 22, 5: 43, 6: 31, 7: 29, 8: 24}`
- component_shape_match: `265/265`
- component_exactish_any: `8/265`
- max_color_crop_top_left: `0/265`
- largest_crop_top_left: `0/265`
- any_same_shape_input_crop_exact: `5/265`

## 判断

task130はshapeだけならcomponent familyだが、出力内容は単純なcomponent crop/maskやmax-color/largest proxyでは説明できない。短期のcost probe対象にはせず、別候補へpivotする。

## リスク

- leakage risk: low。入力/出力構造監査のみ。
- overfitting risk: medium-low。shape一致だけで内容ruleを推定すると過学習しやすい。

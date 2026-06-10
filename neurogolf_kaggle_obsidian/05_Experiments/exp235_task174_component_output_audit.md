# exp235_task174_component_output_audit

## 目的

task174がtask300同様、最大componentまたは最大nonzero色count cropで解けるか監査する。

## 結果

- baseline cost: `52019`
- largest_crop_pass: `138/266`
- max_color_crop_pass: `138/266`
- max_color_matches_largest: `266/266`

## 判断

task174は最大色/最大componentの選択までは簡単だが、出力はcomponent全体cropではない。内部sub-cropまたはshape ruleの監査が必要。

## リスク

- leakage risk: low
- overfitting risk: medium-low

# exp235_task174_component_output_audit

## 目的

task174がtask300同様、最大componentまたは最大nonzero色count cropで解けるか監査する。

## 結果

- baseline_cost: `52019`
- largest_crop_pass: `138/266`
- max_color_crop_pass: `138/266`
- max_color_matches_largest: `266/266`

## 判断

max_color_cropがfullならtask300 loweringを拡張する。そうでなければ失敗patternを監査する。

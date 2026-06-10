# exp245_task100_static_template_audit

## 目的

task100は2x2固定・binary signature 1種類のstable-binary候補なので、templateと色selectorを監査する。

## 結果

- baseline_cost: `6536`
- output_shape: `2x2`
- template: `(1, 1, 1, 1)`
- best color rule: `mode_nz` / `max_count_low` = `248/266`

## 判断

mode色でかなり近い。残り18例の切替条件を監査する。

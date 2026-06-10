# exp250_task242_static_template_color_audit

## 目的

task242は3x3固定・binary template 1種類・baseline `20119` なので、色selectorをcount/bbox/component特徴で監査する。

## 結果

- template: 3x3 all-nonzero `266/266`
- output_color_hist: 1-9へ分散
- best rule: `count_max_low 64/266`
- target_count_rankも0-8に分散

## 判断

templateは固定だが色selectorが弱い。static 3x3 loweringには進まない。

# exp249_task274_template_selector_audit

## 目的

task274は3x3固定・binary signature 4種類なので、templateと色selectorが単純特徴で決まるか監査する。

## 結果

- baseline_cost: `8581`
- output_color: always `8`
- template_hist: 4種類、`70/69/66/64`
- best template rule: `70/269`

## 判断

色は固定だがtemplate selectorが未解決。static 3x3 loweringには進まない。

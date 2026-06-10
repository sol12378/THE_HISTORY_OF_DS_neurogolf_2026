# exp208_task185_bg_distribution_audit

## 目的

task185 selector scoreで背景色を除外する必要があるため、背景色分布を確認する。

## 結果

- bg_hist: `{1: 32, 2: 23, 3: 30, 4: 25, 5: 26, 6: 26, 7: 34, 8: 37, 9: 34}`
- 27x27 / 29x29 の両shapeで背景色は分散。

## 判断

固定bg maskは使えない。task185を続けるなら、入力から動的に背景色を推定してselector/coreから除外する必要がある。

## リスク

- leakage risk: low。入力分布監査のみ。
- overfitting risk: medium-low。固定bg採用は過学習になる。

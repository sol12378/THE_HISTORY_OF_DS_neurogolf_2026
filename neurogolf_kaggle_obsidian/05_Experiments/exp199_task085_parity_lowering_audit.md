# exp199_task085_parity_lowering_audit

## 目的

task085のsolved ruleを安いONNX loweringにできるか、まず固定global checkerboard parityで近似可能か確認する。

## 結果

- relative_rule_pass: `265/265`
- global_parity_pass: `{0: 33, 1: 40}`
- left_parity_hist: `{0: 435, 1: 375}`
- width_hist: `{29: 34, 13: 91, 9: 66, 11: 59, 7: 65, 5: 92, 19: 68, 17: 69, 15: 89, 25: 33, 27: 35, 3: 59, 23: 26, 21: 24}`

## 判断

固定global checkerboardではtask085の「bar左端から見た偶奇」を表現できない。短いONNX loweringにはrun-left parity検出、またはbar内部のrelative positionを別経路で作る必要がある。

## リスク

- leakage risk: low。入力構造の監査のみ。
- overfitting risk: low。候補artifactやsubmissionは生成していない。

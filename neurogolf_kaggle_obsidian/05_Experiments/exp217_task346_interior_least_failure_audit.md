# exp217_task346_interior_least_failure_audit

## 目的

task346 `interior_least_nz` の2失敗を監査し、componentなしの安い補正で救えるか確認する。

## 結果

- pass/fail: `265/2`
- fail idx: `89`, `95`
- simple count/edge/corner predicateでTP2/FP0は見つからず。

## 判断

task346はまだcomponent-like conditionが必要。exp215のcost wallから直接loweringは保留。

## リスク

- leakage risk: low
- overfitting risk: medium

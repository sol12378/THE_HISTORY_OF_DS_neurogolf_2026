# exp072_task251_predicate_probe

## 目的

`exp071` で最優先候補になった `task251` について、単純なchanged-cell predicateでどこまで説明できるかを測る。

## 結果

- `inside_ray4_adj1plus`: TP 2708 / FP 113 / FN 0
- `reject_straight_adj2`: FPは消える方向だがFNが出るため不採用

## 解釈

task251は `inside nonzero bbox + 4方向rayで非ゼロに挟まれる + 隣接非ゼロあり` で全positiveを拾える。残り113 false positiveだけを削る問題に縮んだ。

ただしFP signatureはsingletonが多く、raw signature table化は高risk。次はray distanceとlocal component contextを使った小branch treeを合成する。

## リスク

- leakage risk: low。提出物なし。
- overfitting risk: medium。signature memorizationを禁止し、説明可能なbranchのみ採用する。

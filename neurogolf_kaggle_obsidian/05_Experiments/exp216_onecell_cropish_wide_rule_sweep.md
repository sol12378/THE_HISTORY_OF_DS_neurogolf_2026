# exp216_onecell_cropish_wide_rule_sweep

## 目的

exp087の1x1 cropish候補全体へ、安くloweringできる可能性のある入力集約ruleを横展開する。

## 結果

- evaluated: `48/56/103/291/346/355`
- full_hits: none
- near_hit: task346 `interior_least_nz` `265/267`

## 判断

1x1単純aggregateだけでは提出候補なし。task346は改善したが、残り2例の補正が必要。

## リスク

- leakage risk: low
- overfitting risk: medium-low

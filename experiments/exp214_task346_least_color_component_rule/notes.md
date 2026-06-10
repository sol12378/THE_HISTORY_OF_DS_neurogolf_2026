# exp214_task346_least_color_component_rule

## 目的

task346の1x1 outputについて、least-color近似の4failをcomponent特徴で補正したruleをfull validationする。

## 結果

- pass: `267/267`
- branch_hist: `{'rank0': 263, 'rank1': 4}`

## 判断

Python ruleはfull-pass。ただしONNX化には色countとlargest connected component proxyが必要で、即tiny loweringではない。

## リスク

- leakage risk: low。
- overfitting risk: medium。threshold 8 は4 switch例から得たためhiddenで要注意。

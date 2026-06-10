# exp214_task346_least_color_component_rule

## 目的

task346のleast-color近似をcomponent特徴で補正したruleをfull validationする。

## 結果

- pass: `267/267`
- branch_hist: `{rank0: 263, rank1: 4}`
- rule: 非zero色のcount最少色を選ぶ。ただしその色の最大4-connected componentが8以上ならcount2位を選ぶ。

## 判断

task346は新しいsolved-rule asset。ただしONNX化には色countとlargest component size proxyが必要で、即提出候補ではない。

## リスク

- leakage risk: low。入力構造rule。
- overfitting risk: medium。threshold 8は4 switch例から導出。

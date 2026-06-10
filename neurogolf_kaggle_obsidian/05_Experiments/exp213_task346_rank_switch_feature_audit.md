# exp213_task346_rank_switch_feature_audit

## 目的

task346のrank0/rank1切替を説明する構造特徴を探す。

## 結果

- switch_count: `4`
- `rank0_largest_component >= 8` が TP4 / FP0 / FN0
- `rank1_largest_component <= 3` も TP4 / FP0 / FN0

## 判断

最大連結成分サイズで4 switchを説明できる。補正规則としてfull validationする。

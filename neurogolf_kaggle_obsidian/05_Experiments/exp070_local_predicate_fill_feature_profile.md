# exp070_local_predicate_fill_feature_profile

## 目的

`exp069` LOCAL_PREDICATE_FILL上位10 taskを、changed-cell構造から分類し、task020型の小template selectorが使えるか確認する。

## 結果

- targets: `173, 77, 158, 133, 71, 285, 85, 251, 383, 54`
- class counts:
  - `complex_sparse_or_object_edit`: 10
- promising simple template tasks: none

## 解釈

上位10 taskは、task020のような少数template sparse fillではなかった。多くは変更セル数が大きく、target colorも単純ではなく、component/object/color-role editとして扱う必要がある。

task251だけは `0 -> color1` のfillで比較的近いが、canonical template数が84あり、固定template selectorでは足りない。

## Decision

次はcomponent/color-role profilerへ切り替える。具体的には、connected component、border/hole、object color、target/background role、row/column/ray構造を抽出し、変更セルを説明するpredicate treeを作る。

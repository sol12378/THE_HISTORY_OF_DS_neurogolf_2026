# exp070_local_predicate_fill_feature_profile

## 目的

`exp069` LOCAL_PREDICATE_FILL_COMPILER上位taskを、changed-cell feature profileで分類する。

## 結果

- targets: [173, 77, 158, 133, 71, 285, 85, 251, 383, 54]
- class counts: {'complex_sparse_or_object_edit': 10}
- promising tasks: []

## Decision

promising taskがあれば次にselector/rule minerへ進む。なければcomponent/color-role profileへ深掘りする。

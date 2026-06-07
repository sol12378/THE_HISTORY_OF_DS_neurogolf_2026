# exp080_task366_marker_object_copy_rule

## 目的

`task366` をsource panel object と target panel marker の対応として説明できるか検証する。

## 結果

- candidate: `object_marker_copy_by_color_shape`
- validation: `266/266`
- train: `3/3`
- test: `1/1`
- arc-gen: `262/262`
- mean cell accuracy: `1.0000`

## 判断

full passなら、次はobject-marker copy ruleを低cost ONNXへ落とすcompiler設計へ進む。
partialなら、marker groupingまたはsource object matchingを改善する。

## Risk

- leakage risk: low-to-medium。raw lookupではなくobject/marker構造rule。
- overfitting risk: medium。lowering前にall-arc full passが必要。

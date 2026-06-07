# exp077_task366_object_anchor_crop_profile

## 目的

最大gain候補の `task366` について、低cost `Slice` / `Gather` へ落とせるcrop ruleがあるかを調べる。

## 結果

- examples: `266`
- signal counts: `{'no_crop_signal': 266}`
- shape pair count: `60`
- train exact table eval: `0/266` (`no_table`)

## 判断

この実験はprofileのみで、ONNXは生成していない。
exact crop / nonzero bbox / component bbox / color-remap crop の信号を見て、次のobject-anchor crop compilerを絞る。

## Risk

- leakage risk: low-to-medium。train tableは全arc-genで評価するが、raw tableをそのまま提出しない。
- overfitting risk: medium。shape-position tableはbranch圧縮が必要。

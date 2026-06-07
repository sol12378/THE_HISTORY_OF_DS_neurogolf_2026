# exp_b020_l4_shape_crop_pattern_profiler

## 目的

L4 shape/crop lane top taskについて、固定anchor crop / shape-anchor table / nonzero bbox cropで説明できるかをprofileする。

## 結果

- target tasks: `10`
- lower candidate count: `0`
- tested patterns:
  - fixed anchor crop: `tl`, `tr`, `bl`, `br`, `center`
  - train shape-to-anchor table
  - nonzero bbox crop
- all top10 decision: `not_simple_crop_anchor`
- submission: `no_submit`

## 解釈

L4上位は名前上はshape/cropだが、単純な固定Sliceやnonzero bbox cropでは説明できない。object-anchor crop、shape transform、resize/remap、またはobject role selectionが必要。

## Risk

- leakage risk: low-to-medium。trainでtableを推定し、all arc-genで評価したのみ。
- overfitting risk: medium。次のobject-anchor ruleはfull arc-gen gate必須。

## Decision

L4の単純固定crop routeは下げる。次はobject-anchor crop profile、または提出候補に近いfull-arc-safe graph surgery delta / L3 object move laneへ寄せる。

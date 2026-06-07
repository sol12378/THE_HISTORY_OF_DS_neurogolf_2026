# exp_b027_task251_closed_zero_component_rule

## 目的

`exp070` で有望に見えた task251 を、説明可能な sparse/region fill rule に圧縮できるか確認する。

## 結果

- full pass rule: `closed_zero_component_neighbor2_to_1`
- validation: `266/266`
- rule: 0-component がgrid borderへ接しておらず、4-neighbor境界色が `2` だけなら、そのcomponentを color `1` にする。
- cheap row/column approximation: `195/266`

## 判断

task251は説明可能ruleとして解けた。ただし、素朴なflood-fill unrollは過去実験で高cost化しやすい。次はclosed component maskを安くONNX化する lowering 専用実験に進む。

## Risk

- leakage risk: low-to-medium: rule is simple and explanatory; verified on all arc-gen examples but not yet hidden LB.
- overfitting risk: medium: task-specific component rule needs ONNX validation and LB delta before adoption.

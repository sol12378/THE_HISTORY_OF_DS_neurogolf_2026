# exp081_task366_lowering_inventory

## 目的

`exp080` のtask366 full-pass ruleをONNX化する前に、static template detectorで小さく表現できるか棚卸しする。

## 結果

- examples: `266`
- unique source templates: `695`
- common used templates: `511`
- max source objects/example: `3`
- max used objects/example: `3`
- max marker cells: `7`
- estimated total used kernel params: `7014`

## 判断

この実験はinventoryのみで、ONNXは生成していない。
template数とkernel proxyが小さければ、次はcorrectness-first static detectorを生成する。
大きすぎる場合は、object shapeを個別templateではなく構造ruleへ圧縮する。

## Risk

- leakage risk: low-to-medium。all arc-genでcompiler sizeを測るが、raw lookup提出物は生成しない。
- overfitting risk: medium。template列挙のままではhidden汎化が弱い可能性がある。

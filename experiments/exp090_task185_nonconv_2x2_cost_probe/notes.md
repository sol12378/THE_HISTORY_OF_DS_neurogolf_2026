# exp090_task185_nonconv_2x2_cost_probe

## 目的

`task185` のhomogeneous 2x2 block判定をConvなしで表し、`exp089` のcore proxy cost `1147` を下げられるか測る。

## 結果

`result.json` と `cost_probe.csv` を参照。

## Decision

600を超える場合は、標準ONNXでの新規loweringより既存artifact surgeryや、よりfusedな表現を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice位置のproxyで、提出候補ではない。

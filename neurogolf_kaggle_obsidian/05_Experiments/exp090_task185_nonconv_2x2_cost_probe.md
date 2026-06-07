# exp090_task185_nonconv_2x2_cost_probe

## 目的

`task185` のhomogeneous 2x2 block判定をConvなしで表し、`exp089` のcore proxy cost `1147` を下げられるか測る。

## 結果

- `mul4_static_lattice`: cost `2848`
- `mul4_direct_shifted_slices`: cost `2200`
- baseline cost: `59584`

## 解釈

one-hotの4 shifted viewsを `Mul` で重ねる表現は、paramsは小さいが中間3x3 tensorが多く、公式memory costが支配的になる。grouped Conv proxy `1147` より悪い。

## Decision

task185はrule hitとして保持するが、標準ONNX新規loweringで600級を狙う優先度を下げる。次は既存artifact surgery、または4 shifted Slice / Mulをさらにfuseできる表現を探す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice coordinate proxyであり提出候補ではない。

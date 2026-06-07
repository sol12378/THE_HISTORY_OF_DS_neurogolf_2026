# exp096_one_node_template_replacements

## 目的

`exp095` のfull-hit one-node templateを公式ONNX replacementとして生成し、現在のbest bundleへ差し替え可能か評価する。

## 結果

- evaluated candidates: `4`
- accepted tasks: `[]`
- accepted delta: `0.0`
- status: `no_gain`

全候補が `0_pass_1_fail` でrejected。

## 解釈

Python gridではfull hitだったが、公式one-hotではpadded 30x30全体に対してflip/rot180をかけるため、実grid外のpaddingが移動してactive regionと一致しなくなる。つまり、単純1ノードfull-grid transformは、可変サイズARC gridのpadding semanticsと合わない。

## Decision

zip由来の1ノード公式テンプレートは、full 30x30 active grid、またはpaddingを動かさないshape-aware transformに限定して使う。一般の可変サイズflip/rotは crop -> transform -> pad が必要で、1ノード極小costにはならない。

## Risk

- leakage risk: low。
- overfitting risk: low。

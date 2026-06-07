# exp096_one_node_template_replacements

## 目的

`exp095` のfull-hit one-node templateを公式ONNX replacementとして生成し、現在のbest bundleへ差し替え可能か評価する。

## 結果

- evaluated candidates: `4`
- accepted tasks: `[]`
- accepted delta: `0.000000`
- new local estimate: `6282.812218`

## Decision

改善がある場合は、full validation済みmicro-deltaとしてsubmit候補にする。rot90は2ノード中間が重いので今回除外。

## Risk

- leakage risk: low。
- overfitting risk: low。

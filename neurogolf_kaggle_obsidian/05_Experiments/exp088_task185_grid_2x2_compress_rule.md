# exp088_task185_grid_2x2_compress_rule

## 目的

`exp087` のP0 cropish最上位 `task185` の説明可能ruleを確認する。

## 結果

- pass: `267/267`
- fail: `0`
- baseline cost: `59584`

## Rule

入力は背景格子色を持つ大きなlatticeで、格子交点上に4x4の色付きmatrixがある。この4x4 matrixについて、隣接2x2 blockが非zero同色で完全に埋まる場合だけ、その色を3x3出力の対応位置へ置く。それ以外は0。

## Decision

task185をONNX lowering候補に昇格する。固定3x3出力なので600級の可能性はあるが、4x4 lattice抽出と2x2同色判定のcost gateを先に測る。

## Risk

- leakage risk: low。geometry ruleでありraw lookupではない。
- overfitting risk: medium-low。all arc-gen pass済みだが、ONNX化で絶対座標に過依存しない。

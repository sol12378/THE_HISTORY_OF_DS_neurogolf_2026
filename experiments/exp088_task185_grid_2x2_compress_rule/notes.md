# exp088_task185_grid_2x2_compress_rule

## 目的

`exp087` のP0 cropish最上位 `task185` について、固定3x3出力の説明可能ruleを確認する。

## 仮説

入力は格子線上に4x4の色付きmatrixを持つ。隣接2x2 blockが非zero同色で完全に埋まる場合だけ、その色を3x3出力の対応位置に置く。

## 結果

- pass: `267/267`
- fail: `0`
- baseline cost: `59584`

## Decision

full passなら、次はONNX loweringへ進む。出力は固定3x3なので、`exp086` の3x3 `Slice+Pad` proxy `381` の近傍を狙える。ただし格子線/4x4抽出と2x2同色判定がselector costを追加するため、まずcost proxyを測る。

## Risk

- leakage risk: low。geometry ruleであり、raw lookupではない。
- overfitting risk: medium-low。all arc-genで確認済みだが、ONNX化では絶対座標/周期に過依存しない。

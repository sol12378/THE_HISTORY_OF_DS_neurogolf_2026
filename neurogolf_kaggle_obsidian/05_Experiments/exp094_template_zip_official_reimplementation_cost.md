# exp094_template_zip_official_reimplementation_cost

## 目的

zipテンプレートは直接公式互換ではないが、同じ幾何変換を公式one-hot形式で再実装した場合に250〜600級へ入るかを測る。

## 結果

- `official_identity`: cost `0`
- `official_flip_lr`: cost `4`
- `official_rot180`: cost `8`
- `official_transpose_hw`: cost `0`
- `official_channel_gather_recolor`: cost `10`
- `official_rot90_ccw`: cost `36004`

## 解釈

公式one-hotでも、1ノードの `Slice` / `Transpose` / channel `Gather` は非常に安い。一方、`Transpose+Slice` のように30x30中間tensorを作る2ノード構成は、memory costが一気に支配的になる。

## Decision

zipはtemplate taxonomyとして使う。特に「1ノードで表せるか」「中間30x30を作らないか」を判定軸にする。uint8/int-gridの実測costはそのまま採用しない。

## Risk

- leakage risk: low。
- overfitting risk: low。

# exp078_task366_axis_halving_probe

## 目的

`task366` が2-row/2-col pair reductionで説明できるか確認する。

## 結果

- candidates: `36`
- best: `rows_min`
- exact pass: `0/266`
- mean cell accuracy: `0.2336`

## 判断

隣接pair圧縮ではない。片軸半分のshapeはpanel構造によるものと判断する。

## Risk

- leakage risk: low。
- overfitting risk: low。

# exp079_task366_panel_overlay_probe

## 目的

`task366` が上下/左右2パネルの単純overlayで説明できるか確認する。

## 結果

- candidates: `22`
- best: `vertical_min`
- exact pass: `0/266`
- mean cell accuracy: `0.2850`

## 判断

単純overlayではない。source panelのobjectとtarget panelのmarker対応を明示する必要がある。

## Risk

- leakage risk: low。
- overfitting risk: low。

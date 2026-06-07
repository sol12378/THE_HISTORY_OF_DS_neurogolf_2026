# exp077_task366_object_anchor_crop_profile

## 目的

最大gain候補 `task366` が単純crop/object-anchor crop/color-remap cropで説明できるか調べる。

## 結果

- examples: `266`
- exact crop signal: `0/266`
- train exact crop table: none
- signal counts: all `no_crop_signal`
- output shapeは入力の片軸半分になるが、単純cropではない。

## 判断

naive `Slice` crop routeは不採用。panel分割・object marker copy系へ進む。

## Risk

- leakage risk: low。診断のみ。
- overfitting risk: low。候補採用なし。

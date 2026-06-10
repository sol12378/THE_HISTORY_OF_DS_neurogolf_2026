# exp222_fixed_smallshape_crop_colormap_sweep

## 目的

固定小shape crop候補にglobal color-mapを重ね、static Slice + LUTで表現できる候補を探す。

## 結果

- full hits: task039/task135/task326のみ
- useful fixed hits: none
- full hitのcolor-mapはすべてidentity

## 判断

global color-mapは新規supplier gainを生まなかった。fixed small-shape crop/colormap laneはいったん主力から下げる。

## リスク

- leakage risk: medium-low。診断として全available examples由来のmapを見ている。
- overfitting risk: medium。

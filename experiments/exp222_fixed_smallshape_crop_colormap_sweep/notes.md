# exp222_fixed_smallshape_crop_colormap_sweep

## 目的

固定小shape crop候補に、task共通のglobal color-mapを重ねることで、static Slice + LUTで表現できる候補を探す。

## 結果

- full_hits: `[(39, '3x3', 'bbox_bottom_left_dr0_dc0_rot270', 7772), (135, '3x3', 'top_right_dr0_dc0_id', 360), (326, '2x2', 'bbox_top_left_dr0_dc0_id', 160)]`
- fixed_anchor_full_hits: `[(135, '3x3', 'top_right_dr0_dc0_id', 360)]`
- useful_fixed_hits(cost>600): `[]`
- near_hits: `[]`

## 判断

useful fixed-anchor full hitがあればcost probeへ進める。全例由来のcolor-mapなので、提出前にはtrain-derived mapで再検証する。

## リスク

- leakage risk: medium-low。診断として全available examplesからmapを見ている。
- overfitting risk: medium。

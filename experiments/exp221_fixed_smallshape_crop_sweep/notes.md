# exp221_fixed_smallshape_crop_sweep

## 目的

固定小shape cropish候補に、static Slice化しやすい固定anchor cropと、診断用のbbox cropを横展開する。

## 結果

- evaluated: `[6, 22, 38, 39, 48, 56, 79, 100, 103, 111, 121, 130, 134, 135, 146, 149, 153, 185, 207, 235, 242, 263, 271, 274, 291, 296, 316, 326, 334, 346, 347, 355, 386, 391, 393, 395, 399]`
- full_hits: `[(39, '3x3', 'bbox_bottom_left_dr0_dc0_rot270'), (135, '3x3', 'top_right_dr0_dc0_id'), (326, '2x2', 'bbox_top_left_dr0_dc0_id')]`
- fixed_anchor_full_hits: `[(135, '3x3', 'top_right_dr0_dc0_id')]`
- near_hits: `[]`

## 判断

fixed-anchor full hitがあればstatic Slice cost probeへ進める。bbox-only full hitはexp219/220のcost wallを踏まえて慎重に扱う。

## リスク

- leakage risk: low。
- overfitting risk: medium-low。

# exp218_crop3x3_wide_rule_sweep

## 目的

exp087の3x3固定cropish候補に、安いSlice/transformでloweringできるcrop ruleを横展開する。

## 結果

- evaluated: `[6, 22, 39, 79, 111, 121, 130, 134, 135, 146, 149, 153, 185, 235, 242, 263, 271, 274, 296, 316, 334, 347, 395, 399]`
- full_hits: `[(39, 'bbox_bottom_left_dr0_dc0_rot270'), (135, 'top_right_dr0_dc0_id')]`
- near_hits(<=10 fail): `[]`
- best: `[(6, 'bottom_left_dr-1_dc-1_anti_transpose', 8, 266), (22, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 266), (39, 'bbox_bottom_left_dr0_dc0_rot270', 264, 264), (79, 'bbox_top_left_dr0_dc0_id', 47, 266), (111, 'bbox_top_left_dr1_dc0_id', 160, 265), (121, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 266), (130, 'center_dr-1_dc-1_id', 2, 265), (134, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 266), (135, 'top_right_dr0_dc0_id', 266, 266), (146, 'bbox_top_left_dr0_dc0_id', 96, 267), (149, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 267), (153, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 265), (185, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 267), (235, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 69), (242, 'bbox_bottom_left_dr-1_dc0_rot270', 9, 266), (263, 'bbox_bottom_right_dr0_dc0_id', 72, 267), (271, 'bbox_bottom_left_dr0_dc0_id', 31, 267), (274, 'bbox_top_left_dr1_dc1_flipud', 54, 269), (296, 'bbox_bottom_left_dr0_dc-1_fliplr', 1, 268), (316, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 266), (334, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 271), (347, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 269), (395, 'bbox_top_left_dr-1_dc-1_anti_transpose', 1, 268), (399, 'bbox_bottom_left_dr-1_dc-1_anti_transpose', 0, 273)]`

## 判断

full hitはSlice/transform cost probeへ進める。なければ単純3x3 crop laneは縮小し、色変換やmask抽出を含むfamilyへ移る。

## リスク

- leakage risk: low。入力だけのcrop/transform。
- overfitting risk: medium-low。offset sweepが広いため、full hit以外は補正追加に注意。

# exp223_small_output_mask_shape_inventory

## 目的

小出力cropish候補について、出力shapeが入力bbox/component/color-countなどの簡単なshape特徴で説明できるか、またbinary mask/templateとして安定しているかを棚卸しする。

## 結果

- task_count: `58`
- shape_full_hits(top): `[(300, 77546, 'one_component_shape', '2x3:12;3x2:7;3x3:161;4x3:87'), (174, 52019, 'one_component_shape', '1x2:4;1x5:1;2x1:6;2x2:31;2x3:42;2x4:28;2x5:34;3x1:3;3x2:28;3x3:19;3x4:30;4x1:1;4x2:13;4x3:14;5x1:1;5x2:11'), (130, 15948, 'one_component_shape', '3x3:265'), (355, 11795, 'one_component_shape', '1x1:267'), (346, 9178, 'one_component_shape', '1x1:267'), (393, 459, 'num_colors_by_1', '3x1:265'), (56, 213, 'num_colors_by_1', '1x1:46'), (103, 193, 'num_colors_by_1', '1x1:223')]`
- stable_binary_high_cost(top): `[(253, 48269, 1, '4x4:265'), (355, 11795, 1, '1x1:267'), (391, 10021, 3, '3x1:267'), (346, 9178, 1, '1x1:267'), (100, 6536, 1, '2x2:266'), (48, 5609, 1, '1x1:270'), (291, 4327, 1, '1x1:265')]`

## 判断

shape_full_hitsはshape-branch supplier候補、stable_binary_high_costはmask/template supplier候補として個別監査する。提出なし。

## リスク

- leakage risk: low。
- overfitting risk: low。

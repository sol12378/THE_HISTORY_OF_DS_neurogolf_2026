# exp209_task365_bbox_branch_inventory

## 目的

task365のselected bbox/shape分布を確認し、少数branch loweringで押せる余地があるか判断する。

## 結果

- fail_count: `0`
- unique_selected_bbox_count: `186`
- selected_shape_hist: `{'(3, 3)': 47, '(4, 3)': 34, '(3, 4)': 27, '(5, 3)': 25, '(4, 4)': 21, '(5, 4)': 19, '(3, 6)': 19, '(3, 5)': 17, '(6, 3)': 13, '(4, 6)': 12, '(5, 5)': 9, '(4, 5)': 8, '(6, 4)': 7, '(6, 5)': 4, '(5, 6)': 3, '(6, 6)': 1}`
- selected_rank_hist: `{0: 106, 1: 99, 2: 61}`
- count2_margin_hist: `{1: 174, 2: 72, 3: 20}`

## 判断

bbox/shape空間が広い場合、task365のdynamic object selectionには深入りせず、より小さいcropish taskへpivotする。

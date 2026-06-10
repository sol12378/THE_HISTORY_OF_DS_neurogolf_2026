# exp209_task365_bbox_branch_inventory

## 目的

task365のselected bbox/shape分布を確認し、少数branch loweringで押せる余地があるか判断する。

## 結果

- fail_count: `0`
- unique_selected_bbox_count: `186`
- unique_object_signature_count: `266`
- selected_rank_hist: `{0: 106, 1: 99, 2: 61}`

## 判断

task365はbranch-table化に向かない。dynamic object selectionが必要で、現時点では別taskへpivotする。

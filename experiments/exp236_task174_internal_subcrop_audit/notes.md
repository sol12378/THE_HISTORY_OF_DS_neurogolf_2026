# exp236_task174_internal_subcrop_audit

## 目的

task174について、最大色component bbox/maskの内部sub-cropで出力を説明できるか監査する。

## 結果

- best_rules: `[{'rule': 'bbox_bottom_center', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_bottom_center_fliplr', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_bottom_left', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_bottom_left_fliplr', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_bottom_right', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_bottom_right_fliplr', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_center', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_center_fliplr', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_center_left', 'pass_count': 138, 'fail_count': 128}, {'rule': 'bbox_center_left_fliplr', 'pass_count': 138, 'fail_count': 128}]`
- full_hits: `[]`
- near_hits: `[]`
- examples_without_hit_count: `128`

## 判断

full hitがあればcost probeへ進む。なければtask174はより複雑な内部選択が必要。

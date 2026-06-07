# exp_b020_l4_shape_crop_pattern_profiler

## 目的

L4 shape/crop lane top taskについて、固定anchor crop / shape-anchor table / nonzero bbox cropで説明できるかをprofileする。

## 結果

- target tasks: 10
- lower candidates: 0
- top profiles: [{'task_id': 366, 'strict_cost': 836880, 'gain_to_250': 8.115975052155397, 'train_examples': 3, 'total_examples': 266, 'input_shape_count': 60, 'output_shape_count': 30, 'train_crop_any_pass': 0, 'train_anchor_rule': '{}', 'train_anchor_pass': 0, 'train_anchor_total': 3, 'full_anchor_pass': 0, 'full_anchor_total': 266, 'bbox_crop_pass': 0, 'bbox_crop_total': 266, 'shape_to_anchor_table': '{}', 'decision': 'not_simple_crop_anchor'}, {'task_id': 158, 'strict_cost': 182776, 'gain_to_250': 6.594555720492073, 'train_examples': 3, 'total_examples': 266, 'input_shape_count': 33, 'output_shape_count': 33, 'train_crop_any_pass': 0, 'train_anchor_rule': '{}', 'train_anchor_pass': 0, 'train_anchor_total': 3, 'full_anchor_pass': 0, 'full_anchor_total': 266, 'bbox_crop_pass': 0, 'bbox_crop_total': 266, 'shape_to_anchor_table': '{}', 'decision': 'not_simple_crop_anchor'}, {'task_id': 77, 'strict_cost': 157913, 'gain_to_250': 6.448338609581491, 'train_examples': 3, 'total_examples': 266, 'input_shape_count': 18, 'output_shape_count': 18, 'train_crop_any_pass': 0, 'train_anchor_rule': '{}', 'train_anchor_pass': 0, 'train_anchor_total': 3, 'full_anchor_pass': 0, 'full_anchor_total': 266, 'bbox_crop_pass': 0, 'bbox_crop_total': 266, 'shape_to_anchor_table': '{}', 'decision': 'not_simple_crop_anchor'}, {'task_id': 396, 'strict_cost': 143389, 'gain_to_250': 6.351855577835492, 'train_examples': 3, 'total_examples': 266, 'input_shape_count': 49, 'output_shape_count': 24, 'train_crop_any_pass': 0, 'train_anchor_rule': '{}', 'train_anchor_pass': 0, 'train_anchor_total': 3, 'full_anchor_pass': 0, 'full_anchor_total': 266, 'bbox_crop_pass': 0, 'bbox_crop_total': 266, 'shape_to_anchor_table': '{}', 'decision': 'not_simple_crop_anchor'}, {'task_id': 64, 'strict_cost': 114153, 'gain_to_250': 6.123834014959286, 'train_examples': 4, 'total_examples': 267, 'input_shape_count': 168, 'output_shape_count': 168, 'train_crop_any_pass': 0, 'train_anchor_rule': '{}', 'train_anchor_pass': 0, 'train_anchor_total': 4, 'full_anchor_pass': 0, 'full_anchor_total': 267, 'bbox_crop_pass': 0, 'bbox_crop_total': 267, 'shape_to_anchor_table': '{}', 'decision': 'not_simple_crop_anchor'}]

## 判断

full passのstatic shape-anchor候補があればSlice loweringへ進む。なければobject-anchor crop profileへ進む。

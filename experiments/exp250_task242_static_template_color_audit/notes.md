# exp250_task242_static_template_color_audit

## 目的

task242は3x3固定・binary template 1種類・baseline高めなので、色selectorをcount/bbox/component特徴で監査する。

## 結果

- baseline_cost: `20119`
- template_hist: `[((1, 1, 1, 1, 1, 1, 1, 1, 1), 266)]`
- output_color_hist: `{1: 32, 2: 29, 3: 30, 4: 34, 5: 30, 6: 26, 7: 23, 8: 32, 9: 30}`
- target_count_rank_hist: `{0: 64, 1: 52, 2: 48, 3: 37, 4: 28, 5: 18, 6: 9, 7: 8, 8: 2}`
- target_area_rank_hist: `{0: 37, 1: 28, 2: 27, 3: 41, 4: 31, 5: 34, 6: 29, 7: 26, 8: 13}`
- best_rules: `[{'rule': 'count_max_low', 'pass_count': 64, 'fail_count': 202}, {'rule': 'count_rank0', 'pass_count': 64, 'fail_count': 202}, {'rule': 'count_max_high', 'pass_count': 62, 'fail_count': 204}, {'rule': 'largest_component_max_low', 'pass_count': 57, 'fail_count': 209}, {'rule': 'largest_component_max_high', 'pass_count': 53, 'fail_count': 213}, {'rule': 'count_rank1', 'pass_count': 52, 'fail_count': 214}, {'rule': 'count_rank2', 'pass_count': 48, 'fail_count': 218}, {'rule': 'area_rank3', 'pass_count': 41, 'fail_count': 225}, {'rule': 'area_rank0', 'pass_count': 37, 'fail_count': 229}, {'rule': 'bbox_area_max_low', 'pass_count': 37, 'fail_count': 229}]`

## 判断

単純color selectorがfullならstatic 3x3 ONNX probeへ進む。弱ければ保留。

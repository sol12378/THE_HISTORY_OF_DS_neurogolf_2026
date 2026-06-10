# exp248_task291_onecell_selector_audit

## 目的

task291は1x1固定・binary signature 1種類なので、出力色がcount/bbox/componentなどの単純selectorで決まるか監査する。

## 結果

- baseline_cost: `4327`
- output_color_hist: `{1: 29, 2: 33, 3: 31, 4: 28, 5: 26, 6: 24, 7: 31, 8: 26, 9: 37}`
- target_count_rank_hist: `{0: 37, 1: 61, 2: 76, 3: 48, 4: 27, 5: 14, 6: 2}`
- best_rules: `[{'rule': 'count_rank2', 'pass_count': 76, 'fail_count': 189}, {'rule': 'r1_max_high', 'pass_count': 63, 'fail_count': 202}, {'rule': 'edge_count_min_low', 'pass_count': 62, 'fail_count': 203}, {'rule': 'count_rank1', 'pass_count': 61, 'fail_count': 204}, {'rule': 'r0_max_low', 'pass_count': 58, 'fail_count': 207}, {'rule': 'bbox_area_max_low', 'pass_count': 57, 'fail_count': 208}, {'rule': 'bbox_area_max_high', 'pass_count': 56, 'fail_count': 209}, {'rule': 'edge_count_min_high', 'pass_count': 56, 'fail_count': 209}, {'rule': 'r1_max_low', 'pass_count': 56, 'fail_count': 209}, {'rule': 'c1_max_low', 'pass_count': 54, 'fail_count': 211}]`

## 判断

単純selectorがfullなら1x1 ONNX cost probeへ進む。なければ保留。

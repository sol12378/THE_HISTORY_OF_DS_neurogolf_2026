# exp245_task100_static_template_audit

## 目的

task100は2x2固定・binary signature 1種類のstable-binary候補なので、出力templateと色selectorを監査する。

## 結果

- baseline_cost: `6536`
- template_hist: `[((1, 1, 1, 1), 266)]`
- output_color_hist: `{1: 33, 2: 39, 3: 35, 4: 27, 5: 18, 6: 16, 7: 42, 8: 24, 9: 32}`
- best_color_rules: `[{'rule': 'max_count_low', 'pass_count': 248, 'fail_count': 18}, {'rule': 'mode_nz', 'pass_count': 248, 'fail_count': 18}, {'rule': 'max_count_high', 'pass_count': 244, 'fail_count': 22}, {'rule': 'bbox_tl', 'pass_count': 115, 'fail_count': 151}, {'rule': 'bbox_bl', 'pass_count': 109, 'fail_count': 157}, {'rule': 'bbox_tr', 'pass_count': 95, 'fail_count': 171}, {'rule': 'bbox_center', 'pass_count': 88, 'fail_count': 178}, {'rule': 'center', 'pass_count': 83, 'fail_count': 183}, {'rule': 'bbox_br', 'pass_count': 77, 'fail_count': 189}, {'rule': 'bl', 'pass_count': 58, 'fail_count': 208}]`

## 判断

単純color ruleがfullならstatic 2x2 ONNXへ進む。なければ保留。

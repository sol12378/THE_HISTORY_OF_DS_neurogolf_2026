# exp246_task100_mode_failure_audit

## 目的

task100のmode_nz rule `248/266` の残り18例を監査し、count rankや色bbox特徴で切替条件が作れるか確認する。

## 結果

- baseline_cost: `6536`
- mode_low: `248/266`
- target_rank_hist: `{0: 248, 1: 18}`
- best_rules: `[{'rule': 'bbox_area_max', 'pass_count': 266, 'fail_count': 0}, {'rule': 'count_max', 'pass_count': 248, 'fail_count': 18}, {'rule': 'mode_low', 'pass_count': 248, 'fail_count': 18}, {'rule': 'rank0', 'pass_count': 248, 'fail_count': 18}, {'rule': 'mode_high', 'pass_count': 244, 'fail_count': 22}, {'rule': 'bbox_w_max', 'pass_count': 207, 'fail_count': 59}, {'rule': 'bbox_h_max', 'pass_count': 190, 'fail_count': 76}, {'rule': 'c0_min', 'pass_count': 170, 'fail_count': 96}, {'rule': 'r0_min', 'pass_count': 160, 'fail_count': 106}, {'rule': 'on_edge_max', 'pass_count': 141, 'fail_count': 125}]`

## 判断

単純selectorがfullならstatic 2x2 ONNXへ進む。なければtask100は保留。

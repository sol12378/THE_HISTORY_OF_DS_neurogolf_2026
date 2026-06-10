# exp249_task274_template_selector_audit

## 目的

task274は3x3固定・binary signature 4種類なので、templateと色selectorが単純特徴で決まるか監査する。

## 結果

- baseline_cost: `8581`
- template_hist: `[((1, 1, 1, 0, 0, 0, 0, 0, 0), 70), ((1, 1, 1, 0, 0, 1, 0, 0, 0), 69), ((1, 0, 0, 0, 0, 0, 0, 0, 0), 66), ((1, 1, 0, 0, 0, 0, 0, 0, 0), 64)]`
- output_color_hist: `{8: 269}`
- best_color_rules: `[{'rule': 'center', 'pass_count': 186, 'fail_count': 83}, {'rule': 'least_nz', 'pass_count': 186, 'fail_count': 83}, {'rule': 'mode_nz', 'pass_count': 63, 'fail_count': 206}, {'rule': 'bbox_w', 'pass_count': 50, 'fail_count': 219}, {'rule': 'bbox_h', 'pass_count': 45, 'fail_count': 224}, {'rule': 'mode_all', 'pass_count': 2, 'fail_count': 267}, {'rule': 'bbox_area', 'pass_count': 0, 'fail_count': 269}, {'rule': 'bbox_bl', 'pass_count': 0, 'fail_count': 269}, {'rule': 'bbox_br', 'pass_count': 0, 'fail_count': 269}, {'rule': 'bbox_tl', 'pass_count': 0, 'fail_count': 269}]`
- best_template_rules: `[{'rule': 'bl', 'pass_count': 70, 'fail_count': 199}, {'rule': 'br', 'pass_count': 70, 'fail_count': 199}, {'rule': 'mode_all', 'pass_count': 70, 'fail_count': 199}, {'rule': 'tl', 'pass_count': 70, 'fail_count': 199}, {'rule': 'tr', 'pass_count': 70, 'fail_count': 199}, {'rule': 'color_count', 'pass_count': 66, 'fail_count': 203}, {'rule': 'center', 'pass_count': 25, 'fail_count': 244}, {'rule': 'bbox_area', 'pass_count': 0, 'fail_count': 269}, {'rule': 'bbox_bl', 'pass_count': 0, 'fail_count': 269}, {'rule': 'bbox_br', 'pass_count': 0, 'fail_count': 269}]`

## 判断

template/color selectorがfullならstatic 3x3 probeへ進む。弱ければ保留。

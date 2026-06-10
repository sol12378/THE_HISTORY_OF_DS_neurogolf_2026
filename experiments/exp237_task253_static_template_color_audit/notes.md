# exp237_task253_static_template_color_audit

## 目的

task253は4x4固定binary templateに見えるため、出力色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline_cost: `48269`
- binary_signature_count: `1`
- best_color_rules: `[{'rule': 'least_nz', 'pass_count': 76, 'fail_count': 189}, {'rule': 'mode_nz', 'pass_count': 76, 'fail_count': 189}, {'rule': 'bbox_tl', 'pass_count': 29, 'fail_count': 236}, {'rule': 'bbox_bl', 'pass_count': 28, 'fail_count': 237}, {'rule': 'bbox_tr', 'pass_count': 22, 'fail_count': 243}, {'rule': 'bl', 'pass_count': 6, 'fail_count': 259}, {'rule': 'bbox_center', 'pass_count': 4, 'fail_count': 261}, {'rule': 'tl', 'pass_count': 4, 'fail_count': 261}, {'rule': 'center', 'pass_count': 3, 'fail_count': 262}, {'rule': 'tr', 'pass_count': 2, 'fail_count': 263}]`

## 判断

simple color ruleがfullならstatic template ONNXへ進む。なければcolor selectorを別途監査する。

# exp244_task391_static_template_audit

## 目的

task391は3x1固定出力・binary signature 3種類のstable-binary候補なので、出力templateと色が入力の簡単な特徴で決まるか監査する。

## 結果

- baseline_cost: `10021`
- binary_signature_count: `1`
- binary_hist: `[((1, 1, 1), 267)]`
- output_color_hist: `{1: 23, 2: 36, 3: 34, 4: 26, 5: 21, 6: 38, 7: 31, 8: 29, 9: 29}`
- best_color_rules: `[{'rule': 'bbox_bl', 'pass_count': 49, 'fail_count': 218}, {'rule': 'bbox_br', 'pass_count': 43, 'fail_count': 224}, {'rule': 'bbox_tl', 'pass_count': 43, 'fail_count': 224}, {'rule': 'bbox_tr', 'pass_count': 36, 'fail_count': 231}, {'rule': 'bbox_center', 'pass_count': 18, 'fail_count': 249}, {'rule': 'center', 'pass_count': 18, 'fail_count': 249}, {'rule': 'bl', 'pass_count': 0, 'fail_count': 267}, {'rule': 'br', 'pass_count': 0, 'fail_count': 267}, {'rule': 'least_nz', 'pass_count': 0, 'fail_count': 267}, {'rule': 'mode_all', 'pass_count': 0, 'fail_count': 267}]`
- best_template_rules: `[{'rule': 'bl', 'pass_count': 267, 'fail_count': 0}, {'rule': 'br', 'pass_count': 267, 'fail_count': 0}, {'rule': 'mode_all', 'pass_count': 267, 'fail_count': 0}, {'rule': 'tl', 'pass_count': 267, 'fail_count': 0}, {'rule': 'tr', 'pass_count': 267, 'fail_count': 0}, {'rule': 'bbox_center', 'pass_count': 170, 'fail_count': 97}, {'rule': 'center', 'pass_count': 170, 'fail_count': 97}, {'rule': 'bbox_bl', 'pass_count': 0, 'fail_count': 267}, {'rule': 'bbox_br', 'pass_count': 0, 'fail_count': 267}, {'rule': 'bbox_tl', 'pass_count': 0, 'fail_count': 267}]`

## 判断

単純selectorがfullならstatic 3x1 ONNXへ進む。なければ別候補へpivotする。

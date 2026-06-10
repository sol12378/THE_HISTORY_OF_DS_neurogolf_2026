# exp240_task271_crop_offset_selector_audit

## 目的

task271は出力が全例で入力内の3x3 exact cropだったため、そのoffsetを単純な入力特徴で選べるか監査する。

## 結果

- baseline_cost: `28991`
- exact_offset_count_hist: `{1: 267}`
- top_exact_offsets: `[('6,6', 20), ('0,1', 17), ('0,6', 16), ('0,0', 16), ('1,6', 15), ('6,0', 14), ('0,5', 13), ('6,1', 12), ('1,0', 11), ('0,2', 11)]`
- best_selectors: `[{'selector': 'comp_area_desc_clamped', 'hit': 176, 'miss': 91, 'within1': 176}, {'selector': 'comp_area_desc_tl', 'hit': 176, 'miss': 91, 'within1': 176}, {'selector': 'comp_size_desc_clamped', 'hit': 106, 'miss': 161, 'within1': 106}, {'selector': 'comp_size_desc_tl', 'hit': 106, 'miss': 161, 'within1': 106}, {'selector': 'comp_color_asc_clamped', 'hit': 86, 'miss': 181, 'within1': 92}, {'selector': 'comp_color_asc_tl', 'hit': 85, 'miss': 182, 'within1': 92}, {'selector': 'comp_r0_asc_clamped', 'hit': 68, 'miss': 199, 'within1': 68}, {'selector': 'comp_r0_asc_tl', 'hit': 68, 'miss': 199, 'within1': 68}, {'selector': 'comp_c0_asc_clamped', 'hit': 63, 'miss': 204, 'within1': 63}, {'selector': 'comp_c0_asc_tl', 'hit': 63, 'miss': 204, 'within1': 63}]`

## 判断

fullまたはnear-fullのselectorがあればPython rule化とcost probeへ進む。なければtask271はcrop offset selector未解決として保留する。

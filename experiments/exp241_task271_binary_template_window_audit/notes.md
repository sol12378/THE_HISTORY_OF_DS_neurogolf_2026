# exp241_task271_binary_template_window_audit

## 目的

task271の出力は固定binary signatureを持つ3x3 exact cropだったため、固定binary/color-signature templateに一致する入力windowを選べるか監査する。

## 結果

- baseline_cost: `28991`
- binary_template: `(1, 1, 1, 1, 1, 1, 1, 1, 1)`
- signature_template: `(2, 1, 2, 1, 2, 1, 1, 2, 1)`
- counters: `{'target_in_binary_cands': 267, 'target_in_signature_cands': 1, 'binary_unique_and_target': 0, 'signature_unique_and_target': 1}`
- candidate_count_hists: `{'binary_4': 267, 'signature_0': 263, 'signature_1': 4}`
- best_rank_selectors: `[{'selector': 'binary_closest_tl', 'hit': 73, 'eligible': 267, 'miss': 194}, {'selector': 'binary_first_row_major', 'hit': 68, 'eligible': 267, 'miss': 199}, {'selector': 'binary_first_col_major', 'hit': 63, 'eligible': 267, 'miss': 204}, {'selector': 'binary_last_row_major', 'hit': 63, 'eligible': 267, 'miss': 204}, {'selector': 'binary_last_col_major', 'hit': 60, 'eligible': 267, 'miss': 207}, {'selector': 'binary_closest_br', 'hit': 58, 'eligible': 267, 'miss': 209}, {'selector': 'signature_closest_br', 'hit': 1, 'eligible': 1, 'miss': 266}, {'selector': 'signature_closest_tl', 'hit': 1, 'eligible': 1, 'miss': 266}, {'selector': 'signature_first_col_major', 'hit': 1, 'eligible': 1, 'miss': 266}, {'selector': 'signature_first_row_major', 'hit': 1, 'eligible': 1, 'miss': 266}]`

## 判断

template一致windowが一意、または単純rankで選べるならrule/cost probeへ進む。複数候補が多くrankも弱い場合はselectorを追加で掘る。

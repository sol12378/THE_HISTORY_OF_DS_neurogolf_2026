# exp242_task271_block_color_selector_audit

## 目的

task271は各入力に3x3 full-nonzero blockが4個あり、そのうち1個が出力と一致する。4候補から正解blockを色構成rankで選べるか監査する。

## 結果

- baseline_cost: `28991`
- best_selectors: `[{'selector': 'color8_count_min', 'hit': 267, 'miss': 0}, {'selector': 'sum_colors_min', 'hit': 267, 'miss': 0}, {'selector': 'edge_sum_min', 'hit': 204, 'miss': 63}, {'selector': 'corner_sum_min', 'hit': 188, 'miss': 79}, {'selector': 'center_min', 'hit': 139, 'miss': 128}, {'selector': 'least_count_min', 'hit': 126, 'miss': 141}, {'selector': 'distinct_max', 'hit': 75, 'miss': 192}, {'selector': 'min_color_min', 'hit': 75, 'miss': 192}, {'selector': 'mode_count_max', 'hit': 74, 'miss': 193}, {'selector': 'max_color_max', 'hit': 68, 'miss': 199}]`
- target_feature_hists: `{'signature_freq': [((1, 1, 1, 2, 1, 1, 1, 1, 1), 17), ((1, 1, 1, 1, 1, 1, 1, 2, 1), 17), ((1, 1, 2, 1, 1, 1, 1, 1, 1), 16), ((1, 1, 1, 1, 1, 1, 1, 1, 2), 16), ((1, 1, 1, 1, 1, 2, 1, 1, 1), 13)], 'count_pattern': [((8, 1), 122), ((7, 2), 61), ((6, 3), 47), ((5, 4), 37)], 'position': [('6,6', 20), ('0,1', 17), ('0,6', 16), ('0,0', 16), ('1,6', 15)], 'distinct': [(2, 267)], 'mode_count': [(8, 122), (7, 61), (6, 47), (5, 37)], 'color8_count': [(1, 122), (2, 61), (3, 46), (4, 24), (5, 13)]}`

## 判断

単純rank/color特徴がfullまたはnear-fullならPython selector化する。弱い場合はtask271を一旦保留し、別候補へpivotする。

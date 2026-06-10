# exp206_task185_dynamic_output_diagnostic

## 目的

exp204 dynamic candidateのexample 0 mismatchをtensor diffで診断する。

## 結果

- variants: `['dynamic_axis_basic', 'dynamic_axis_nonzero_only', 'dynamic_axis_score_nonzero_core_basic', 'dynamic_axis_score_nonzero_default_bg']`
- equal: `{'dynamic_axis_basic': False, 'dynamic_axis_nonzero_only': False, 'dynamic_axis_score_nonzero_core_basic': False, 'dynamic_axis_score_nonzero_default_bg': False}`
- diff_count: `{'dynamic_axis_basic': 8, 'dynamic_axis_nonzero_only': 9, 'dynamic_axis_score_nonzero_core_basic': 11, 'dynamic_axis_score_nonzero_default_bg': 12}`

## 判断

diff patternに応じて、bg-channel suppression、出力位置、selector-index correctionのどれを次に直すか決める。

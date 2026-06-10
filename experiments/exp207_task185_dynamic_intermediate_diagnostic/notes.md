# exp207_task185_dynamic_intermediate_diagnostic

## 目的

exp204 dynamic candidateの中間テンソルを出力化し、example 0でONNXの`row_idx`/`col_idx`/`lattice_4x4`がPython selectorと一致するか確認する。

## 結果

- python_rr: `[5, 8, 11, 14]`
- python_cc: `[5, 8, 11, 14]`
- row_match: `{'dynamic_axis_basic': False, 'dynamic_axis_score_nonzero_core_basic': False, 'dynamic_axis_score_nonzero_default_bg': False}`
- col_match: `{'dynamic_axis_basic': False, 'dynamic_axis_score_nonzero_core_basic': False, 'dynamic_axis_score_nonzero_default_bg': False}`
- output_equal: `{'dynamic_axis_basic': False, 'dynamic_axis_score_nonzero_core_basic': False, 'dynamic_axis_score_nonzero_default_bg': False}`

## 判断

indexが一致していればcore/output mappingを修正する。不一致ならArgMax-to-template mappingかwindow orderingを修正する。

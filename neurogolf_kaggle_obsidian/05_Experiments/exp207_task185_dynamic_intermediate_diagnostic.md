# exp207_task185_dynamic_intermediate_diagnostic

## 目的

exp204 dynamic candidateの中間テンソルを出力化し、example 0でONNXの`row_idx`/`col_idx`/`lattice_4x4`がPython selectorと一致するか確認する。

## 結果

- Python bg: `4`
- Python rr/cc: `[5, 8, 11, 14]`
- `dynamic_axis_basic`: rr/cc `[0, 3, 6, 9]`
- `dynamic_axis_score_nonzero_core_basic`: rr/cc `[2, 5, 8, 11]`
- `dynamic_axis_score_nonzero_default_bg`: rr/cc `[2, 5, 8, 11]`

## 判断

ONNX selector indexがPython selectorと一致していない。color0除外だけでは背景色4を除外できず、背景線がselector scoreに入っている。

## リスク

- leakage risk: low。validation exampleの中間診断。
- overfitting risk: medium。example 0診断のみ。

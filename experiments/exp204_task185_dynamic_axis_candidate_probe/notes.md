# exp204_task185_dynamic_axis_candidate_probe

## 目的

task185で、dilated axis selectorから`GatherElements`用index tensorを作り、4x4 lattice extraction + homogeneous coreまで接続したcorrectness-first candidateを検証する。

## 結果

- variants: `['dynamic_axis_basic', 'dynamic_axis_nonzero_only', 'dynamic_axis_score_nonzero_core_basic', 'dynamic_axis_score_nonzero_default_bg']`
- accepted_count: `0`
- validation_status: `{'dynamic_axis_basic': '0_pass_1_fail', 'dynamic_axis_nonzero_only': '0_pass_1_fail', 'dynamic_axis_score_nonzero_core_basic': '0_pass_1_fail', 'dynamic_axis_score_nonzero_default_bg': '0_pass_1_fail'}`
- candidate_status: `{'dynamic_axis_basic': 'rejected', 'dynamic_axis_nonzero_only': 'rejected', 'dynamic_axis_score_nonzero_core_basic': 'rejected', 'dynamic_axis_score_nonzero_default_bg': 'rejected'}`
- cost: `{'dynamic_axis_basic': 106720, 'dynamic_axis_nonzero_only': 142730, 'dynamic_axis_score_nonzero_core_basic': 142730, 'dynamic_axis_score_nonzero_default_bg': 143490}`

## 判断

full validation passなら提出候補。failなら最初のmismatchから、bg/special maskまたはselector-score maskを追加して次実験へ進む。

## リスク

- leakage risk: low。
- overfitting risk: medium。task-specific geometryを使うが、raw coordinate tableではない。

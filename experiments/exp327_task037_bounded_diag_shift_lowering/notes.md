# exp327_task037_bounded_diag_shift_lowering

## 目的

task037 `opposite_ray_diag_same_color` rule を、full-grid Conv visibility ではなく bounded diagonal shift stack で lowering して cost gain が出るか確認する。

## 結果

- base cost: `63726`
- candidate validation: `266_pass_0_fail`
- candidate status: `no_cost_gain`
- candidate cost: `6633317`
- local delta: `0.0`

## 判断

no_submit: validation failed or no cost gain

## Risk

- leakage risk: low: explicit task037 diagonal ray rule using input-only shifts; no public/private labels or output lookup.
- overfitting risk: medium: task-specific full-arc rule; if improved, Kaggle single-task calibration is still needed.

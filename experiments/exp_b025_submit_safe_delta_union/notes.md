# exp_b025_submit_safe_delta_union

## 目的

Kaggle転移が確認済みのsubmit-safe deltaを合成する。

- base: `exp068_seddik_style_strict_scalarization`
- overlay: `exp_b021_strict_seed_exp038_micro_delta_submit`

## 結果

- accepted tasks: [62, 145, 255, 268]
- local delta over exp068: 0.201867334
- new local estimate: 6282.812217709
- submission decision: submit_for_union_delta_lb_calibration

## Risk

- leakage risk: low-to-medium: union of two already submitted full-arc-safe strict-derived deltas.
- overfitting risk: medium: task-specific graph surgery remains small but should be LB-calibrated after union.

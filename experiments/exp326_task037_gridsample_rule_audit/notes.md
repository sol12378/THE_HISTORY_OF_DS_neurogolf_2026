# exp326_task037_gridsample_rule_audit

## 目的

exp325 採点待ち中の独立作業。task037 が task251 より GridSample/shift lowering に向くか、rule と data profile を再監査する。

## 結果

- rule: `opposite_ray_diag_same_color`
- validation: `266_pass_0_fail`
- changed cells mean: `8.466165413533835`
- output new color examples: `0`
- generated new color cells: `0`
- max endpoint distance hist: `{1: 255, 2: 544, 3: 697, 4: 506, 5: 250}`

## 判断

task037 remains a plausible GridSample/shift lowering target: the rule is full-arc valid, outputs introduce no new nonzero colors, and each changed cell copies an existing diagonal endpoint color. Next implementation should avoid full-grid Conv visibility and instead test a bounded diagonal shift stack or GridSample-generated diagonal samples with cheap equality masks.

## Submission

`no_submit: audit only; no ONNX replacement generated.`

## Risk

- leakage risk: low: revalidates an explanatory rule on local task examples only.
- overfitting risk: low-to-medium: task-specific rule is full-arc validated, but a future ONNX candidate still needs cost and Kaggle calibration.

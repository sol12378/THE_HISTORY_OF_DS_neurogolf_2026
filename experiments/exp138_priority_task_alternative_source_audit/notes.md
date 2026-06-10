# exp138_priority_task_alternative_source_audit

## Hypothesis

private failure consensus上位taskには、既存実験内に代替sourceまたはfull-validation済み候補があり、bisection提出前に修復候補を絞れる。

## Result

- targets: `[13, 2, 29, 9, 24, 18]`
- manifest rows: `203`
- candidate eval rows: `199`

## Task Summary

- task013: sources `12`, non-current `22`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `116882.0`
- task002: sources `12`, non-current `22`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `97293.0`
- task029: sources `11`, non-current `20`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `97117.0`
- task009: sources `11`, non-current `20`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `81241.0`
- task024: sources `8`, non-current `8`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `9080.0`
- task018: sources `12`, non-current `22`, full hints `4`, best_alt `exp004_public_blend_relaxed_static` from `exp004_public_blend_relaxed_static` cost `114578.0`

## Decision

Validate non-current alternatives for the six consensus tasks first; if none are robust, build a bisection probe group from task013/002/029/009.

## Leakage / Overfitting Risk

low: scans existing manifests/candidate evals only; no new model is adopted.
medium: alternative sources may share the same hidden failure mode and require full validation plus LB calibration.

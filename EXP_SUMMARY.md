# Experiment Summary

## Current Best

| Type | Experiment | CV | LB | Notes |
|---|---|---:|---:|---|
| Baseline | exp001_baseline | task087 local all pass | 14.50 | single task087 rot180 ONNX |
| Best CV | exp001_baseline | task087 local all pass | 14.50 | single-task local validation baseline |
| Best LB | exp001_baseline | task087 local all pass | 14.50 | Kaggle submission ref 53383536 |
| Best Local Bundle | exp002_public_blend_6500_fast | local estimate 6282.08 | - | public artifact blend, not submitted |

## Experiments

| Exp | Date | Hypothesis | CV | LB | Decision | Notes |
|---|---|---|---:|---:|---|---|
| exp001_baseline | 2026-06-05 | Establish a reproducible NeuroGolf baseline and first ONNX submission loop. | task087 local all pass | 14.50 | submitted_complete | Official spec confirmed; task087 rot180 ONNX submitted via Kaggle API, ref 53383536. |
| exp002_public_blend_6500_fast | 2026-06-05 | Public artifact blend can reach local 6500 under strict static/cost validation. | local estimate 6282.08, 400/400 selected, 400 sample-pass | - | below_target | Built `submission.zip` locally; 16346 candidates, strict banned-op/static filter. 6500未達。次はrule再監査またはtop expensive task rewrite。 |

# exp_b021_strict_seed_exp038_micro_delta_submit

## 目的

strict seedにexp038 full-arc-safe graph surgeryの4 taskだけを載せ、小さいlocal/LB較正提出候補を作る。

## 結果

- delta tasks: [62, 145, 255, 268]
- validation all ok: True
- local estimate delta: 0.201867
- new local estimate: 6282.432095
- submission decision: submit_for_micro_delta_lb_calibration

## Risk

- leakage risk: low-to-medium: graph surgery deltas are full-arc validated and based on strict seed, but still task-specific.
- overfitting risk: medium: small delta should be Kaggle-calibrated before scaling.

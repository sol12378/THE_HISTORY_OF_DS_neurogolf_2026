# exp144_task018_b035_repair_probe

## Hypothesis

task018の残るfull-ok別rawであるexp_b035候補なら、public-zeroを修復できる可能性がある。

## Result

- status: `public_lb_improved`
- validation: `266_pass_0_fail`
- candidate_cost: `476184`
- expected_lb_if_public_pass: `5942.476440386858`
- zip sha256: `2836b97d300ae7ccc19079ff70d8a9354eeff3877385157719acf0609b76ae49`

## Decision

Kaggle ref `53520958` として提出し、public LB は `5942.48`。exp127 `5930.55` から `+11.93` で、expected `+11.92644` と一致した。

task018 public-zero repairとして採用候補。ただし exp_b035 public/source blend 由来であり、private robustness は未証明。current public LB bestとして記録するが、最終候補化には追加のrisk reviewが必要。

## Leakage / Overfitting Risk

high: exp_b035 is a public/source blend candidate; this is a single-task calibration probe.
high: public pass would not prove private robustness.

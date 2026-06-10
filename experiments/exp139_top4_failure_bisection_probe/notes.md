# exp139_top4_failure_bisection_probe

## Hypothesis

task013/002/029/009 が -352 gap のprivate/public scoring failure候補なら、これらを意図的にfailさせたprobeのLB dropから、現public scoringで既に死んでいるかを判別できる。

## Result

- status: `probe_complete`
- targets: `[13, 2, 29, 9]`
- expected_drop_if_all_alive: `54.05675047799333`
- expected_lb_if_all_alive: `5876.493249522006`
- zip sha256: `3256750319f1402b292faa6e56029717bd5d6e144056ea605101e62c7690353f`

## Target Validation

- task013: `0_pass_1_fail`, reason `mismatch example 0`
- task002: `0_pass_1_fail`, reason `mismatch example 0`
- task029: `0_pass_1_fail`, reason `mismatch example 0`
- task009: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

Kaggle ref `53520609` として提出し、public LB は `5876.49`。exp127 best `5930.55` からのdropは `54.06` で、4 taskが全て生きていた場合の期待drop `54.05675` と一致した。

task013/002/029/009 は public scoring では全て寄与している。したがって、これらは「すでに0点になっているgap原因task」ではない。次はこの4 taskを除外し、task024/018/004/019/032/050/016 などの次点consensus groupをprobeまたは個別監査する。

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

# exp140_next4_failure_bisection_probe

## Hypothesis

exp139でtop4 taskがpublic-scoring aliveと判明したため、次点consensus group `024/018/004/019` をfail-stub probeで測る。

## Result

- status: `probe_complete`
- targets: `[24, 18, 4, 19]`
- expected_drop_if_all_alive: `57.01524045998281`
- expected_lb_if_all_alive: `5873.534759540017`
- zip sha256: `535a48f75efae85b9213d8a0fe8953015e093fb5ca9d9c0d5ebb70088123d165`

## Target Validation

- task024: `0_pass_1_fail`, reason `mismatch example 0`
- task018: `0_pass_1_fail`, reason `mismatch example 0`
- task004: `0_pass_1_fail`, reason `mismatch example 0`
- task019: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

Kaggle ref `53520700` として提出し、public LB は `5886.89`。exp127 best `5930.55` からのdropは `43.66` で、4 task all-alive期待drop `57.01524` より `13.355` 小さい。

この差は task018 の点数 `13.35099` と一致する。task024/task004/task019 はpublic scoringで生きており、task018 はexp127時点で既にpublic 0点だった可能性が非常に高い。

次は task018 のrepair候補を最優先で探す。repairに成功すれば LB +13.35 級を回収できる可能性がある。

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

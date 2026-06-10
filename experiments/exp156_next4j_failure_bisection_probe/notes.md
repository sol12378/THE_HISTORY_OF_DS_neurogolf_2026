# exp156_next4j_failure_bisection_probe

## Hypothesis

exp155後の未判定subset頻出上位 `027/082/088/092` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[27, 82, 88, 92]`
- expected_drop_if_all_alive: `57.48546606439621`
- expected_lb_if_all_alive_from_exp127: `5873.064533935604`
- Kaggle ref: `53522565`
- public LB: `5873.06`
- observed_drop_from_exp127: `57.49`
- drop_delta_vs_all_alive: `0.0045`
- interpretation: 観測LBはall-alive期待値と一致。task027/082/088/092にはpublic-zero evidenceなし。
- zip sha256: `0d6888797652eab09fafd27292c658be62862cbd01ef222e91dddd068c734c58`

## Target Validation

- task027: `0_pass_1_fail`, reason `mismatch example 0`
- task082: `0_pass_1_fail`, reason `mismatch example 0`
- task088: `0_pass_1_fail`, reason `mismatch example 0`
- task092: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

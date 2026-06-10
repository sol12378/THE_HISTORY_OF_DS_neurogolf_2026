# exp153_next4g_failure_bisection_probe

## Hypothesis

exp152後の未判定subset頻出上位 `051/049/046/011` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[51, 49, 46, 11]`
- expected_drop_if_all_alive: `57.023091988774816`
- expected_lb_if_all_alive_from_exp127: `5873.526908011226`
- Kaggle ref: `53522084`
- public LB: `5873.53`
- observed_drop_from_exp127: `57.02`
- drop_delta_vs_all_alive: `-0.0031`
- interpretation: 観測LBはall-alive期待値と一致。task051/049/046/011にはpublic-zero evidenceなし。
- zip sha256: `1474725cece09a6117358175e785b6b59b06c9c7c01f63dd361fb3cd2beae5bb`

## Target Validation

- task051: `0_pass_1_fail`, reason `mismatch example 0`
- task049: `0_pass_1_fail`, reason `mismatch example 0`
- task046: `0_pass_1_fail`, reason `mismatch example 0`
- task011: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

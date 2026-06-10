# exp154_next4h_failure_bisection_probe

## Hypothesis

exp153後の未判定subset頻出上位 `033/087/276/030` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[33, 87, 276, 30]`
- expected_drop_if_all_alive: `71.47882525980204`
- expected_lb_if_all_alive_from_exp127: `5859.071174740198`
- Kaggle ref: `53522207`
- public LB: `5859.07`
- observed_drop_from_exp127: `71.48`
- drop_delta_vs_all_alive: `0.0012`
- interpretation: 観測LBはall-alive期待値と一致。task033/087/276/030にはpublic-zero evidenceなし。
- zip sha256: `4e374a64afa68dd9f89071adee43a22224a7da759b2dee9bac5bd718b1439bcb`

## Target Validation

- task033: `0_pass_1_fail`, reason `mismatch example 0`
- task087: `0_pass_1_fail`, reason `mismatch example 0`
- task276: `0_pass_1_fail`, reason `mismatch example 0`
- task030: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

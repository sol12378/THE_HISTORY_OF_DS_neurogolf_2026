# exp149_next4e_failure_bisection_probe

## Hypothesis

exp148後の未判定subset頻出上位 `053/056/067/005` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[53, 56, 67, 5]`
- expected_drop_if_all_alive: `73.813087369845`
- expected_lb_if_all_alive_from_exp127: `5856.7369126301555`
- Kaggle ref: `53521652`
- public LB: `5856.74`
- observed_drop_from_exp127: `73.81`
- drop_delta_vs_all_alive: `-0.0031`
- interpretation: 観測LBはall-alive期待値と一致。task053/056/067/005にはpublic-zero evidenceなし。
- zip sha256: `83be660efa6afae1b4fe69635d82da7b001ecc7f3c0c18a4470e4aefd32ebb2f`

## Target Validation

- task053: `0_pass_1_fail`, reason `mismatch example 0`
- task056: `0_pass_1_fail`, reason `mismatch example 0`
- task067: `0_pass_1_fail`, reason `mismatch example 0`
- task005: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

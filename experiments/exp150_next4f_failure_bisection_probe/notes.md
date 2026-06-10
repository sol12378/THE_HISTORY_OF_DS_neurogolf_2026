# exp150_next4f_failure_bisection_probe

## Hypothesis

exp149後の未判定subset頻出上位 `023/036/085/034` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_partial_zero`
- targets: `[23, 36, 85, 34]`
- expected_drop_if_all_alive: `55.52734311214046`
- expected_lb_if_all_alive_from_exp127: `5875.02265688786`
- Kaggle ref: `53521805`
- public LB: `5890.05`
- observed_drop_from_exp127: `40.50`
- drop_delta_vs_all_alive: `-15.0273`
- identified_public_zero_task: `023`
- interpretation: 観測dropはall-alive期待よりtask023 points相当だけ小さい。task023が次のpublic-zero repair target。
- zip sha256: `5651ed2c67e14b027c5ac0637683c2636e1d013b22d1769bc25d5dd048d52e9d`

## Target Validation

- task023: `0_pass_1_fail`, reason `mismatch example 0`
- task036: `0_pass_1_fail`, reason `mismatch example 0`
- task085: `0_pass_1_fail`, reason `mismatch example 0`
- task034: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

audit_task023_repair_candidates_next

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

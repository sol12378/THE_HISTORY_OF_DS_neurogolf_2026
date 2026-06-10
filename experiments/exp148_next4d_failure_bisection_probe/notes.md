# exp148_next4d_failure_bisection_probe

## Hypothesis

既知alive、task018修復、task025既存source不可を踏まえ、次点group `366/077/064/084` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[366, 77, 64, 84]`
- expected_drop_if_all_alive: `52.78042443088678`
- expected_lb_if_all_alive_from_exp127: `5877.769575569114`
- Kaggle ref: `53521453`
- public LB: `5877.68`
- observed_drop_from_exp127: `52.87`
- drop_delta_vs_all_alive: `0.0896`
- interpretation: 観測LBはall-alive期待値とほぼ一致。task366/077/064/084には大きなpublic-zero evidenceなし。
- zip sha256: `fe7444e708d3f6aa6c913be1322f05e2624d34f975df063e1aaaf233c490feb9`

## Target Validation

- task366: `0_pass_1_fail`, reason `mismatch example 0`
- task077: `0_pass_1_fail`, reason `mismatch example 0`
- task064: `0_pass_1_fail`, reason `mismatch example 0`
- task084: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

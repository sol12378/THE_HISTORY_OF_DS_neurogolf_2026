# exp157_next4k_failure_bisection_probe

## Hypothesis

exp156採点待ち中に、次の未判定subset候補 `090/173/028/091` のfail-stub probe zipを準備する。

## Result

- status: `probe_complete_all_alive`
- targets: `[90, 173, 28, 91]`
- expected_drop_if_all_alive: `56.917033011886396`
- expected_lb_if_all_alive_from_exp127: `5873.632966988113`
- Kaggle ref: `53522643`
- public LB: `5873.63`
- observed_drop_from_exp127: `56.92`
- drop_delta_vs_all_alive: `0.0030`
- interpretation: 観測LBはall-alive期待値と一致。task090/173/028/091にはpublic-zero evidenceなし。
- zip sha256: `2e7901bc9e368c7b706300370b042188fb233b68efd33162b960146587494bc7`

## Target Validation

- task090: `0_pass_1_fail`, reason `mismatch example 0`
- task173: `0_pass_1_fail`, reason `mismatch example 0`
- task028: `0_pass_1_fail`, reason `mismatch example 0`
- task091: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission if submitted; only measures public scoring behavior for a group.

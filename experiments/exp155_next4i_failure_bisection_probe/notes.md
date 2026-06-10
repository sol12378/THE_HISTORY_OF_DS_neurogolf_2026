# exp155_next4i_failure_bisection_probe

## Hypothesis

exp154後の未判定subset頻出上位 `063/022/037/020` のpublic scoring状態をfail-stub probeで測る。

## Result

- status: `probe_complete_all_alive`
- targets: `[63, 22, 37, 20]`
- expected_drop_if_all_alive: `56.94773068259494`
- expected_lb_if_all_alive_from_exp127: `5873.602269317405`
- Kaggle ref: `53522390`
- public LB: `5873.60`
- observed_drop_from_exp127: `56.95`
- drop_delta_vs_all_alive: `0.0023`
- interpretation: 観測LBはall-alive期待値と一致。task063/022/037/020にはpublic-zero evidenceなし。
- zip sha256: `78b0bd9043079923a3f9d552b199f2a89bfa3b1e15f736b7147cfbe106e98583`

## Target Validation

- task063: `0_pass_1_fail`, reason `mismatch example 0`
- task022: `0_pass_1_fail`, reason `mismatch example 0`
- task037: `0_pass_1_fail`, reason `mismatch example 0`
- task020: `0_pass_1_fail`, reason `mismatch example 0`

## Decision

diagnostic_complete_exclude_targets_from_public_zero_suspects

## Leakage / Overfitting Risk

low: deliberate fail-stub probe; no private labels used.
medium: consumes a submission and only measures public scoring behavior for a group.

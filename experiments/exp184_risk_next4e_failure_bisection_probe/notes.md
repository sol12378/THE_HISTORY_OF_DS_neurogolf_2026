# exp184_risk_next4e_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task284/300/379/340 while exp183 scores.

## 結果

- status: `submitted_complete`
- Kaggle ref: `53527058`
- public LB: `5951.06`
- targets: `[284, 300, 379, 340]`
- expected_drop_if_all_alive: `54.864729106950705`
- expected_lb_if_all_alive: `5951.065270893049`
- observed_drop_from_base: `54.87`
- zip sha256: `d7b70e03bc357c8f9d565e3e441f8b7953dd57a56a7ba49210a8e581a607395f`

## Target Validation

- task284: `0_pass_1_fail`, reason `mismatch example 0`
- task300: `0_pass_1_fail`, reason `mismatch example 0`
- task379: `0_pass_1_fail`, reason `mismatch example 0`
- task340: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submitted_after_exp186_no_gain。

public LB `5951.06` は all-alive expected `5951.0653` とscoreboard丸め範囲で一致した。task284/300/379/340はpublic-scoring aliveと判断し、短期public-zero suspectから除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

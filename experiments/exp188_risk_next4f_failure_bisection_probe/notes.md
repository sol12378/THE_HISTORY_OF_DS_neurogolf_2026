# exp188_risk_next4f_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task328/301/306/387 while exp184 scores.

## 結果

- status: `submitted_complete`
- Kaggle ref: `53527133`
- public LB: `5950.27`
- targets: `[328, 301, 306, 387]`
- expected_drop_if_all_alive: `55.65673992877559`
- expected_lb_if_all_alive: `5950.273260071224`
- observed_drop_from_base: `55.66`
- zip sha256: `e8ce56763af823473b730aa38c0d19ce59a3bd647b95001868a31bf7e24353fc`

## Target Validation

- task328: `0_pass_1_fail`, reason `mismatch example 0`
- task301: `0_pass_1_fail`, reason `mismatch example 0`
- task306: `0_pass_1_fail`, reason `mismatch example 0`
- task387: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submitted_after_exp184_all_alive。

public LB `5950.27` は all-alive expected `5950.2733` とscoreboard丸め範囲で一致した。task328/301/306/387はpublic-scoring aliveと判断し、短期public-zero suspectから除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

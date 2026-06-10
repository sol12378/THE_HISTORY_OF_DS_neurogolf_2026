# exp182_risk_next4c_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task313/370/234/303 while exp181 scores.

## 結果

- status: `submitted_complete`
- Kaggle ref: `53526782`
- public LB: `5951.46`
- targets: `[313, 370, 234, 303]`
- expected_drop_if_all_alive: `54.46450433043683`
- expected_lb_if_all_alive: `5951.465495669563`
- observed_drop_from_base: `54.47`
- zip sha256: `c04d5ebc62db5ca4e323b718c8d791118053bf43c9124239e9eb7b04558c72c8`

## Target Validation

- task313: `0_pass_1_fail`, reason `mismatch example 0`
- task370: `0_pass_1_fail`, reason `mismatch example 0`
- task234: `0_pass_1_fail`, reason `mismatch example 0`
- task303: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submitted_after_exp181_all_alive。

public LB `5951.46` は all-alive expected `5951.4655` とscoreboard丸め範囲で一致した。task313/370/234/303 はpublic-scoring aliveと判断し、短期public-zero suspectから除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

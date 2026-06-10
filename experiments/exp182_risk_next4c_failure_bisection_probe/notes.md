# exp182_risk_next4c_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task313/370/234/303 while exp181 scores.

## 結果

- status: `probe_zip_ready`
- targets: `[313, 370, 234, 303]`
- expected_drop_if_all_alive: `54.46450433043683`
- expected_lb_if_all_alive: `5951.465495669563`
- zip sha256: `c04d5ebc62db5ca4e323b718c8d791118053bf43c9124239e9eb7b04558c72c8`

## Target Validation

- task313: `0_pass_1_fail`, reason `mismatch example 0`
- task370: `0_pass_1_fail`, reason `mismatch example 0`
- task234: `0_pass_1_fail`, reason `mismatch example 0`
- task303: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

hold_until_exp181_result

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

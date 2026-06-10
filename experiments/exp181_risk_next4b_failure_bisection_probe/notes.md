# exp181_risk_next4b_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task281/203/126/159 while exp179 scores.

## 結果

- status: `probe_zip_ready`
- targets: `[281, 203, 126, 159]`
- expected_drop_if_all_alive: `54.45312847198618`
- expected_lb_if_all_alive: `5951.476871528014`
- zip sha256: `fca4b54dd97d8a42d0a5a1fd908a0a930847df4ef39cec63e549e471fa4d7757`

## Target Validation

- task281: `0_pass_1_fail`, reason `mismatch example 0`
- task203: `0_pass_1_fail`, reason `mismatch example 0`
- task126: `0_pass_1_fail`, reason `mismatch example 0`
- task159: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

hold_until_exp179_result

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

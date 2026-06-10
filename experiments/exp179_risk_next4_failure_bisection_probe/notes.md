# exp179_risk_next4_failure_bisection_probe

## 目的

risk_score次点の `251/109/239/358` をfail-stub化してpublic-zero有無を測る準備をする。

## 結果

- status: `probe_zip_ready`
- targets: `[251, 109, 239, 358]`
- expected_drop_if_all_alive: `53.9215719532103`
- expected_lb_if_all_alive: `5952.00842804679`
- zip sha256: `c94d4072e3244e935351a1fa6c057e8fe5c8933370166a3a4aac0b2142ec6369`

## Target Validation

- task251: `0_pass_1_fail`, reason `mismatch example 0`
- task109: `0_pass_1_fail`, reason `mismatch example 0`
- task239: `0_pass_1_fail`, reason `mismatch example 0`
- task358: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

hold_until_exp176_result

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

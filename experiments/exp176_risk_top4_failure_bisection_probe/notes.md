# exp176_risk_top4_failure_bisection_probe

## 目的

subset由来候補が尽きた後に備え、risk_score上位の `202/382/205/383` をfail-stub化してpublic-zero有無を測る準備をする。

## 結果

- status: `probe_zip_ready`
- targets: `[202, 382, 205, 383]`
- expected_drop_if_all_alive: `53.291417344804714`
- expected_lb_if_all_alive: `5952.638582655196`
- zip sha256: `0f14355c7e28b39827c7a36852c5624ffda014bdf83f35349adf0383e29e49da`

## Target Validation

- task202: `0_pass_1_fail`, reason `mismatch example 0`
- task382: `0_pass_1_fail`, reason `mismatch example 0`
- task205: `0_pass_1_fail`, reason `mismatch example 0`
- task383: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submit_bisection_probe

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

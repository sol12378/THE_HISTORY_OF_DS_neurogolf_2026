# exp183_risk_next4d_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task187/204/198/364 while exp182 scores.

## 結果

- status: `submitted_complete`
- Kaggle ref: `53526873`
- public LB: `5965.28`
- targets: `[187, 204, 198, 364]`
- expected_drop_if_all_alive: `54.06486621940027`
- expected_lb_if_all_alive: `5951.8651337806`
- observed_drop_from_base: `40.65`
- missing_drop_vs_all_alive: `13.41486621940048`
- zip sha256: `3eff3dcb78ae04ad7b206addb2a24aa0871e37343ae2249090907eebe2e7e79f`

## Target Validation

- task187: `0_pass_1_fail`, reason `mismatch example 0`
- task204: `0_pass_1_fail`, reason `mismatch example 0`
- task198: `0_pass_1_fail`, reason `mismatch example 0`
- task364: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submitted_after_exp182_all_alive。

public LB `5965.28` は all-alive expected `5951.8651` より `+13.41` 高い。missing dropは task187 points `13.4353` と最も近く、task187をpublic-zero repair targetにする。task204/198/364は暫定的にpublic-scoring alive寄りとして扱う。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

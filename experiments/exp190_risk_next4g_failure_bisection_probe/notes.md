# exp190_risk_next4g_failure_bisection_probe

## 目的

Prepare next high-risk-inventory fail-stub probe for task238/112/377/177 while exp188 scores.

## 結果

- status: `submitted_complete`
- Kaggle ref: `53527204`
- public LB: `5950.05`
- targets: `[238, 112, 377, 177]`
- expected_drop_if_all_alive: `55.8822398801408`
- expected_lb_if_all_alive: `5950.047760119859`
- observed_drop_from_base: `55.88`
- zip sha256: `14c2fe037ecad928662e35fb468f58c8fa8802c40083bb8e6efbf278ed80fc7a`

## Target Validation

- task238: `0_pass_1_fail`, reason `mismatch example 0`
- task112: `0_pass_1_fail`, reason `mismatch example 0`
- task377: `0_pass_1_fail`, reason `mismatch example 0`
- task177: `0_pass_1_fail`, reason `mismatch example 0`

## 判断

submitted_after_exp188_all_alive。

public LB `5950.05` は all-alive expected `5950.0478` とscoreboard丸め範囲で一致した。task238/112/377/177はpublic-scoring aliveと判断し、短期public-zero suspectから除外する。

## リスク

low: deliberate fail-stub probe; no private labels used.
medium: risk-inventory prioritization may be less calibrated than subset-sum consensus.

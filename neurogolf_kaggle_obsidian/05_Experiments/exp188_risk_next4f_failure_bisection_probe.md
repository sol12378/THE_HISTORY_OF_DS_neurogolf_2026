# exp188_risk_next4f_failure_bisection_probe

## 結果

- targets: `328/301/306/387`
- status: `submitted_complete`
- Kaggle ref: `53527133`
- public LB: `5950.27`
- expected all-alive LB: `5950.2733`
- zip sanity: pass

## 判断

all-alive expectedと一致。task328/301/306/387は短期public-zero suspectから除外する。

## リスク

fail-stub diagnosticなのでleakage riskは低いが、risk inventory優先度の外挿には不確実性がある。

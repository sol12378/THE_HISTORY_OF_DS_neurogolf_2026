# exp190_risk_next4g_failure_bisection_probe

## 結果

- targets: `238/112/377/177`
- status: submitted complete
- Kaggle ref: `53527204`
- public LB: `5950.05`
- expected all-alive LB: `5950.0478`
- zip sanity: pass

## 判断

all-alive expectedと一致。task238/112/377/177は短期public-zero suspectから除外する。

## リスク

fail-stub diagnosticなのでleakage riskは低いが、risk inventory優先度の外挿には不確実性がある。

# exp182_risk_next4c_failure_bisection_probe

## 結果

- targets: `313/370/234/303`
- Kaggle ref: `53526782`
- public LB: `5951.46`
- expected all-alive LB: `5951.4655`
- 判断: all-alive一致。4 taskは短期public-zero suspectから除外。

## リスク

- leakage risk: low。fail-stub diagnostic。
- overfitting risk: medium。

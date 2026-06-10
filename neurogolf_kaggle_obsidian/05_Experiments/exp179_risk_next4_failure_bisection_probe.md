# exp179_risk_next4_failure_bisection_probe

## 結果

- targets: `251/109/239/358`
- Kaggle ref: `53526669`
- public LB: `5952.01`
- 判断: all-alive expectedと一致。4 taskは短期public-zero suspectから除外。

## リスク

- leakage risk: low。fail-stub diagnostic。
- overfitting risk: medium。risk inventory優先度の較正は限定的。

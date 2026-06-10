# exp181_risk_next4b_failure_bisection_probe

## 結果

- targets: `281/203/126/159`
- Kaggle ref: `53526726`
- public LB: `5951.48`
- 判断: all-alive expectedと一致。4 taskは短期public-zero suspectから除外。

## リスク

- leakage risk: low。fail-stub diagnostic。
- overfitting risk: medium。

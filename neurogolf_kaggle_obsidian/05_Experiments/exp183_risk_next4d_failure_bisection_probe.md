# exp183_risk_next4d_failure_bisection_probe

## 結果

- targets: `187/204/198/364`
- Kaggle ref: `53526873`
- public LB: `5965.28`
- all-alive expected LB: `5951.8651`
- missing drop: `13.41`
- 判断: missing dropがtask187 points `13.4353` と一致。task187をpublic-zero repair targetへ昇格。

## リスク

- 204/198/364はalive寄りだが、単独確認ではないため暫定扱い。

# exp_b034_public6200_notebook_local_estimate

## Hypothesis

ユーザー提供の LB 6200+ Kaggle notebooks は、直接 blend artifact として採用するより、既存の strict/full-arc/LB 較正に通したときの local estimate と必要 delta を見積もる材料として使うべきである。

## Result

現在の submit-safe best は `exp_b025_submit_safe_delta_union`:

- local estimate: `6282.812217709089`
- public LB: `5930.40`
- local - LB gap: `352.4122177090891`

LB 6200 をこの較正 gap で見ると、必要 local は `6552.412217709089`。したがって現状から必要な submit-safe local delta は `+269.60000000000036`。

## Interpretation

5本の blend/baseline 系 notebook は `exp002` 系ですでに代表 source が入っており、直接の新規 local gain は確認できない。新しく実装可能だった情報は Seddik-style surgery と Karnak taxonomy で、これは `exp068` と `exp_b025` により submit-safe local `6282.8122` / LB `5930.40` まで実現済み。

LB 6200+ notebook の leaderboard score は有用な teacher signal だが、public artifact / blend copying は leakage risk と overfitting risk が高い。`exp041` は local `6480.302478` でも LB `3417.71` に崩壊しており、lookup-heavy local upper bound は信用できない。

## Decision

今回の local estimate は `6282.8122` を採用する。LB 6200 を狙うには current calibration 上、あと `+269.6` の full-arc-safe / LB-calibrated local delta が必要。


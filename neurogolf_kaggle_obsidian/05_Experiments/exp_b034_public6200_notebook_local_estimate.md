# exp_b034_public6200_notebook_local_estimate

## 目的

ユーザー提供の LB 6200+ Kaggle notebooks を材料に、現在の local estimate と LB 6200 相当の local target を一度見積もる。

## 結果

- current submit-safe local: `6282.812217709089`
- current submit-safe public LB: `5930.40`
- current local-LB gap: `352.4122177090891`
- LB 6200 equivalent local target: `6552.412217709089`
- current から必要な safe local delta: `+269.60000000000036`

## 判断

blend/baseline 系 notebook は既に `exp002` 系で代表 source が入っている。直接採用による新規 local gain はない。実装済みの有用情報は Seddik-style graph/initializer surgery と Karnak taxonomy で、これは `exp068` / `exp_b025` の submit-safe delta として LB 較正済み。

LB 6200+ notebook は teacher / lowering / taxonomy signal として使う。public blend artifact をそのまま submit-safe estimate に入れない。

## Risk

- leakage risk: direct public blend copying は medium-high
- overfitting risk: notebook artifact reuse は medium-high
- mitigated use: explicit rule / full-arc-safe graph surgery / Kaggle micro-delta calibration


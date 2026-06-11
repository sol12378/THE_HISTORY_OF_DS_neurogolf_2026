# exp321_task243_101_franksunp_repair

## 目的

exp319/exp320 で public-zero repair target とした task101/task243 について、`franksunp_blended_best` の top candidate を exp297 に差し替え、full-arc gate と zip sanity 後に提出する。

## 仮説

task101/task243 が exp297 lineage 上で public-zero なら、full-arc-valid な `franksunp_blended_best` candidate への差し替えで約 `+27.779018` の Public LB 回復が見込める。

## 結果

- status: `submitted_pending`
- task101: validation `266_pass_0_fail`、cost `65875`、candidate points `13.904485714156337`
- task243: validation `265_pass_0_fail`、cost `67878`、candidate points `13.874532744845478`
- zip sanity: `400` files / `names_ok=true`
- expected_public_gain_if_all_fixed: `27.779018459001815`
- expected_public_lb_if_all_fixed: `6036.739018459002`
- Kaggle ref: `53550725`
- Kaggle status: `PENDING`

## 判断

repair bundle として提出済み。採点結果が `6036.74` 付近なら task101/task243 repair 成功として current best 更新。外れた場合は、どちらか片方のみ有効・private/public mismatch・current lineage 解釈ミスを疑い、単体 repair probe に分解する。

## リスク

- leakage risk: medium-high。候補は public-code/raw blend artifact 由来で、private robustness は不確実。
- overfitting risk: medium。target は public diagnostic で特定したが、候補採用は full-arc validation と expected-LB accounting で制御している。

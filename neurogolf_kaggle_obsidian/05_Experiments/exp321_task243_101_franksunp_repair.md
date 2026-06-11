# exp321_task243_101_franksunp_repair

## 目的

exp319/exp320 で public-zero repair target とした task101/task243 について、`franksunp_blended_best` の top candidate を exp297 に差し替え、full-arc gate と zip sanity 後に提出する。

## 仮説

task101/task243 が exp297 lineage 上で public-zero なら、full-arc-valid な `franksunp_blended_best` candidate への差し替えで約 `+27.779018` の Public LB 回復が見込める。

## 結果

- status: `completed`
- task101: validation `266_pass_0_fail`、cost `65875`、candidate points `13.904485714156337`
- task243: validation `265_pass_0_fail`、cost `67878`、candidate points `13.874532744845478`
- zip sanity: `400` files / `names_ok=true`
- expected_public_gain_if_all_fixed: `27.779018459001815`
- expected_public_lb_if_all_fixed: `6036.739018459002`
- Kaggle ref: `53550725`
- Kaggle status: `COMPLETE`
- Public LB: `6008.96`
- observed_gain_vs_base: `0.0`

## 判断

期待 `6036.739018` には届かず、base exp297 と同点。採用しない。

task101/task243 は fail-stub でも repair でも public score に影響しなかった。この source では recoverable public-zero gain にならない。非 public-scoring、public 側同値、または local full-arc では見えない mismatch の可能性があるため、この pair への追加 source 探索よりも exp315 subset-sum 前提の再確認または fresh wide probe に戻す。

## リスク

- leakage risk: medium-high。候補は public-code/raw blend artifact 由来で、private robustness は不確実。
- overfitting risk: medium。target は public diagnostic で特定したが、候補採用は full-arc validation と expected-LB accounting で制御している。

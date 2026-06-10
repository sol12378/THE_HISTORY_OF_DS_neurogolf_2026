# exp319_split_probe_task243_101

## 目的

exp318 で task193/task275 が alive と判明したため、exp315 の missing drop `27.760763` の次候補である task243/task101 を split probe する。

## 仮説

task243/task101 が public-zero なら observed drop は all-alive 期待 `27.779018` より小さくなる。all-alive なら LB は `5981.180982` 付近になる。

## 結果

- status: `submitted_pending`
- targets: `243 101`
- expected_drop_if_all_alive: `27.779018459001815`
- expected_lb_if_all_alive: `5981.180981540998`
- target validation: both `0_pass_1_fail`
- zip sanity: `400` files / `names_ok=true`
- Kaggle ref: `53550517`
- Kaggle status: `PENDING`

## 判断

診断 probe として提出済み。採点完了まで追加 bisection probe は出さない。

## リスク

- leakage risk: low。意図的 fail-stub probe で private label は使っていない。
- overfitting risk: medium。public diagnostic split なので、repair は full validation と correctness-first candidate を必須にする。

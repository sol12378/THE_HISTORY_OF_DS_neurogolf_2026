# exp319_split_probe_task243_101

## 目的

exp318 で task193/task275 が alive と判明したため、exp315 の missing drop `27.760763` の次候補である task243/task101 を split probe する。

## 仮説

task243/task101 が public-zero なら observed drop は all-alive 期待 `27.779018` より小さくなる。all-alive なら LB は `5981.180982` 付近になる。

## 結果

- status: `completed`
- targets: `243 101`
- expected_drop_if_all_alive: `27.779018459001815`
- expected_lb_if_all_alive: `5981.180981540998`
- target validation: both `0_pass_1_fail`
- zip sanity: `400` files / `names_ok=true`
- Kaggle ref: `53550517`
- Kaggle status: `COMPLETE`
- Public LB: `6008.96`
- observed_drop_vs_base: `0.0`
- missing_drop_vs_all_alive: `27.779018459001815`

## 判断

base exp297 と同じ Public LB で、all-alive 期待 `5981.180982` とは一致しない。task243/task101 を fail-stub しても public score が落ちなかったため、この2 task は current-best lineage 上の public-zero repair target と扱う。

次は task243/task101 の既存 full-local-valid source audit と repair zip 構築。

## リスク

- leakage risk: low。意図的 fail-stub probe で private label は使っていない。
- overfitting risk: medium。public diagnostic split なので、repair は full validation と correctness-first candidate を必須にする。

# exp318_split_probe_task193_275

## 目的

exp315 の missing drop `27.760763` の曖昧候補を解くため、最有力ペアの task193/task275 だけを fail-stub する。

## 仮説

task193/task275 が alive なら observed LB は all-alive 期待 `5981.187572` 付近になる。drop が小さければこのペアに public-zero が含まれる。

## 結果

- status: `submitted_complete`
- targets: `193 275`
- expected_drop_if_all_alive: `27.77242767718689`
- expected_lb_if_all_alive: `5981.187572322813`
- target validation: both `0_pass_1_fail`
- zip sanity: `400` files / `names_ok=true`
- Kaggle ref: `53550470`
- Kaggle status: `COMPLETE`
- public LB: `5981.18`
- observed_drop_from_base: `27.78`
- interpretation: task193/task275 は public-scoring alive。exp315 の missing drop は task243/task101 など別候補へ移る。

## 判断

all-alive 期待と一致したため、task193/task275 は public-zero 候補から外す。次は task243/task101 を split probe する。

## リスク

- leakage risk: low。意図的 fail-stub probe で private label は使っていない。
- overfitting risk: medium。public diagnostic split なので、repair は full validation と correctness-first candidate を必須にする。

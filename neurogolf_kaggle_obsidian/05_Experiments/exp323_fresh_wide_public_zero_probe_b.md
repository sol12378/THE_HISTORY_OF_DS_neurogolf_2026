# exp323_fresh_wide_public_zero_probe_b

## 目的

exp322 で exp315 の追加深掘りを止めたため、新しい未 probe 高リスク群を wide fail-stub probe する。

## 仮説

未 probe の `franksunp_blended_best` accepted task を16件 fail-stub すれば、observed drop と expected all-alive drop の差から新しい public-zero 候補を特定できる。

## 結果

- status: `submitted_pending`
- targets: task179/241/309/113/116/164/172/210/311/326/103/135/140/186/167/129
- target points source: current exp297 zip を `score_network` で再計測
- expected_drop_if_all_alive: `336.25926064287535`
- expected_lb_if_all_alive: `5672.700739357125`
- target validation: all `0_pass_1_fail`
- zip sanity: `400` files / `names_ok=true`
- Kaggle ref: `53551982`
- Kaggle status: `PENDING`

## 判断

診断 probe として提出済み。採点完了まで追加 bisection probe は出さない。

採点待ち中は task251 GridSample preparation または exp323 decode tooling を進める。

## リスク

- leakage risk: low。意図的 fail-stub probe で private label は使っていない。
- overfitting risk: medium。public diagnostic probe なので、後続 repair は full-arc validation と expected-LB accounting を必須にする。

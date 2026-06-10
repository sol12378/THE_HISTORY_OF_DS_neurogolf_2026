# exp255_exp254_bypass_bundle_submit_probe

## 目的

exp254でfull-arc passした13件のbypass editsを、current best exp234 bundleへまとめて載せてKaggle提出する。

## 結果

- selected_count: `13`
- selected_tasks: `[133, 204, 216, 54, 280, 205, 382, 239, 398, 275, 158, 234, 19]`
- failures: `[]`
- local_delta: `+0.09024144378056853`
- expected_public_lb_if_calibrated: `6006.41024144378`
- zip_sanity: `400` files, names ok, sha256 `92b3421068f88bd4dc960db3f75b429af873f6af077b43138ef5bbceb752d37c`
- submission: Kaggle ref `53531405`, status COMPLETE, public LB `6006.39`

## 判断

期待値 `6006.4102` に近い public LB `6006.39` を確認。current public bestをexp255へ更新する。

## リスク

- leakage risk: low-medium。full-arc gated graph surgery bundle。
- overfitting risk: medium-low。13 taskすべてavailable examples全通過だが、private側の同値性はLB確認待ち。

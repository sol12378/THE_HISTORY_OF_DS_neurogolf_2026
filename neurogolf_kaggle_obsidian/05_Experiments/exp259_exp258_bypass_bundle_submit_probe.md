# exp259_exp258_bypass_bundle_submit_probe

## 目的

exp258でfull-arc passした23件のrank81-140 bypass editsを、current best exp257 bundleへ積んでKaggle提出する。

## 結果

- selected_count: `23`
- selected_tasks: `[233, 177, 118, 330, 97, 367, 89, 324, 182, 224, 325, 80, 14, 378, 184, 231, 252, 27, 65, 46, 368, 196, 288]`
- failures: `[]`
- local_delta: `+0.21905617266236455`
- expected_public_lb_if_calibrated: `6006.939056172662`
- zip_sanity: `400` files, names ok, sha256 `69281088a38c8ceb6a107195d6b0042dfe4684ac5d9396789a6aaad89bfd63ed`
- submission: Kaggle ref `53531729`, public LB `6006.94`, COMPLETE

## 判断

提出完了。期待値と一致してcurrent public bestへ採用する。

## リスク

- leakage risk: low-medium。full-arc gated graph surgery bundle。
- overfitting risk: medium-low。public LB転写は確認済みだが、private robustnessは引き続き最終統合時に確認する。

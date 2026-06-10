# exp257_exp256_bypass_bundle_submit_probe

## 目的

exp256でfull-arc passした16件のrank31-80 bypass editsを、current best exp255 bundleへ積んでKaggle提出する。

## 結果

- selected_count: `16`
- selected_tasks: `[383, 281, 264, 284, 370, 208, 25, 77, 255, 191, 379, 340, 51, 243, 209, 101]`
- failures: `[]`
- local_delta: `+0.3314725494535331`
- expected_public_lb_if_calibrated: `6006.721472549454`
- zip_sanity: `400` files, names ok, sha256 `eb01cfaa297b7cc671e505f53eda9da78e49a8afd3bf36b16c3265f21ac1b336`
- submission: Kaggle ref `53531558`, status COMPLETE, public LB `6006.72`

## 判断

期待値 `6006.7215` と一致する public LB `6006.72` を確認。current public bestをexp257へ更新する。

## リスク

- leakage risk: low-medium。full-arc gated graph surgery bundle。
- overfitting risk: medium-low。16 task全通過でpublic LB転写は確認済みだが、private robustnessは引き続き未確認。

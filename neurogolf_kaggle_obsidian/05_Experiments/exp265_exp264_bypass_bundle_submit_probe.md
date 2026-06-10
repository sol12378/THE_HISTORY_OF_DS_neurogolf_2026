# exp265_exp264_bypass_bundle_submit_probe

## 目的

exp264でlocal estimateを更新した8件を再生成し、full-arc replayで確認した上でexp263 current bestへstackする。

## 結果

- selected_count: `8`
- failed_count: `0`
- selected_tasks: `299, 229, 235, 339, 129, 144, 318, 26`
- local_delta: `+0.5976596441240858`
- expected_public_lb_if_calibrated: `6008.897659644124`
- zip sanity: `400` files, names ok, sha256 `0c56a04adf6ffd1d0a1ac557f69f3ad39592550c2d57e93501de725612840818`
- Kaggle ref: `53532884`
- status: `COMPLETE`
- public LB: `6008.90`

## 判断

新方針「local estimateを更新した時だけ提出」に従い提出した。public LBは `6008.90` で、期待 `6008.897659644124` と一致しcurrent bestを更新した。

## Risk

- leakage risk: low-medium。replayed full-arc graph surgeryだが、ベースbundleの公開LB較正に依存する。
- overfitting risk: medium-low。rank window一巡後の低cost側deltaなので、LB表示差分が出るかを採点で確認する。

# exp263_exp262_partial_bypass_bundle_submit_probe

## 目的

exp262 partialでlocal estimateを更新した17件を再生成し、full-arc replayで確認した上でexp261 current bestへstackする。

## 結果

- selected_count: `17`
- failed_count: `0`
- selected_tasks: `30, 345, 124, 78, 153, 188, 329, 212, 50, 3, 254, 369, 180, 45, 248, 357, 273`
- local_delta: `+0.5498990065818337`
- expected_public_lb_if_calibrated: `6008.329899006581`
- zip sanity: `400` files, names ok, sha256 `56db2bf23e14c57a141cddaeb4797f03ef205355155dd2f1ae617ae5befac9d9`
- Kaggle ref: `53532720`
- status: `COMPLETE`
- public LB: `6008.30`

## 判断

新方針「local estimateを更新した時だけ提出」に従い提出した。public LBは `6008.30` で、exp261から `+0.52` 改善した。期待 `6008.329899` よりわずかに低いが、graph surgery bundleの較正は引き続き良好。

## Risk

- leakage risk: low-medium。replayed full-arc graph surgeryだが、ベースbundleの公開LB較正に依存する。
- overfitting risk: medium-low。rank window拡張による偶然一致リスクはあるが、これまでのexp255/257/259/261でLB較正は良好。

# exp253_task153_bypass_submit_probe

## 目的

exp252で見つかったtask153 `Reshape_node7_to_input0` bypassを、current public best exp234 bundle上で再生成し、single-task delta提出する。

## 結果

- base_bundle: `experiments/exp234_task300_max_color_submit_probe/submission.zip`
- task153 validation: `265_pass_0_fail`, full-arc ok
- cost: `11212 -> 10947`
- expected_public_lb_if_calibrated: `6006.3439`
- zip_sanity: `400` files, names ok, sha256 `003d03f3aa73f462863f34f36c55dde1db4cbfe69db3d7b7086bc84340fe4621`
- submission: Kaggleへ提出済み。LB確認待ち。

## 判断

微小deltaだが、official-valid delta即提出方針に従って較正する。

## リスク

low-medium。full-arc gated graph surgeryだが、private robustnessはLBで確認する。

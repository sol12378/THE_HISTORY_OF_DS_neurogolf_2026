# exp263_exp262_partial_bypass_bundle_submit_probe

## 目的

exp262 partialでlocal estimateが更新された17件を、exp261 current bestに対して再生成し、full-arc replay gate後にbundle化する。

## 結果

- selected_count: `17`
- failed_count: `0`
- selected_tasks: `[30, 345, 124, 78, 153, 188, 329, 212, 50, 3, 254, 369, 180, 45, 248, 357, 273]`
- local_delta: `0.5498990065818337`
- expected_public_lb_if_calibrated: `6008.329899006581`
- Kaggle ref: `53532720`
- public LB: `6008.30`

## 判断

local estimate更新があり、replayで失敗がなかったため提出した。public LBは `6008.30` でcurrent best更新。

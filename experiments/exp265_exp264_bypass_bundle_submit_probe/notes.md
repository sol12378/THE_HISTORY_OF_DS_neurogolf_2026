# exp265_exp264_bypass_bundle_submit_probe

## 目的

exp264でlocal estimateが更新された8件を、exp263 current bestに対して再生成し、full-arc replay gate後にbundle化する。

## 結果

- selected_count: `8`
- failed_count: `0`
- selected_tasks: `[299, 229, 235, 339, 129, 144, 318, 26]`
- local_delta: `0.5976596441240858`
- expected_public_lb_if_calibrated: `6008.897659644124`

## 判断

local estimate更新があり、replayで失敗がなければ提出対象。

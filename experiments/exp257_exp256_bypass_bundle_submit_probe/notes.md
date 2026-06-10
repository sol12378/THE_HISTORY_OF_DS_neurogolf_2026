# exp257_exp256_bypass_bundle_submit_probe

## 目的

exp256でfull-arc passしたrank31-80 bypass candidatesを、current best exp255 bundleに積んで提出zipを作成する。

## 結果

- selected_count: `16`
- selected_tasks: `[383, 281, 264, 284, 370, 208, 25, 77, 255, 191, 379, 340, 51, 243, 209, 101]`
- failures: `[]`
- local_delta: `0.3314725494535331`
- expected_public_lb_if_calibrated: `6006.721472549454`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 731736, 'sha256': 'eb01cfaa297b7cc671e505f53eda9da78e49a8afd3bf36b16c3265f21ac1b336'}`

## 判断

Kaggle ref `53531558` として提出。public LB `6006.72` でCOMPLETE。

期待 `6006.7215` と一致し、current public bestを更新。以後のbaseはexp257を優先する。

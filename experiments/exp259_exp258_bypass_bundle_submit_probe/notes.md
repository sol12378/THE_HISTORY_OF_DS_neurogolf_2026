# exp259_exp258_bypass_bundle_submit_probe

## 目的

exp258でfull-arc passしたrank81-140 bypass candidatesを、current best exp257 bundleに積んで提出zipを作成する。

## 結果

- selected_count: `23`
- selected_tasks: `[233, 177, 118, 330, 97, 367, 89, 324, 182, 224, 325, 80, 14, 378, 184, 231, 252, 27, 65, 46, 368, 196, 288]`
- failures: `[]`
- local_delta: `0.21905617266236455`
- expected_public_lb_if_calibrated: `6006.939056172662`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 731451, 'sha256': '69281088a38c8ceb6a107195d6b0042dfe4684ac5d9396789a6aaad89bfd63ed'}`

## 判断

Kaggle ref `53531729` として提出済み。public LB `6006.94` でCOMPLETEし、期待値 `6006.939056172662` と一致。current public bestへ採用。

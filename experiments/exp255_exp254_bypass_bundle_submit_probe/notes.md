# exp255_exp254_bypass_bundle_submit_probe

## 目的

exp254でfull-arc passしたtop30 bypass candidatesを、current best exp234 bundleにまとめて載せ、提出zipを作成する。

## 結果

- selected_count: `13`
- selected_tasks: `[133, 204, 216, 54, 280, 205, 382, 239, 398, 275, 158, 234, 19]`
- failures: `[]`
- local_delta: `0.09024144378056853`
- expected_public_lb_if_calibrated: `6006.41024144378`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 731889, 'sha256': '92b3421068f88bd4dc960db3f75b429af873f6af077b43138ef5bbceb752d37c'}`

## 判断

Kaggle ref `53531405` として提出。public LB `6006.39` でCOMPLETE。

期待 `6006.4102` に近い改善が観測され、current public bestを更新。以後のbaseはexp255を優先する。

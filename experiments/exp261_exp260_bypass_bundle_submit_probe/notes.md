# exp261_exp260_bypass_bundle_submit_probe

## 目的

exp260でfull-arc passしたrank141-220 bypass candidatesを、current best exp259 bundleに積んで提出zipを作成する。

## 結果

- selected_count: `27`
- selected_tasks: `[392, 232, 178, 245, 343, 96, 55, 86, 310, 228, 289, 390, 163, 82, 287, 74, 308, 371, 256, 397, 168, 59, 341, 319, 119, 58, 374]`
- failures: `[]`
- local_delta: `0.8400867054431931`
- expected_public_lb_if_calibrated: `6007.780086705443`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 731132, 'sha256': '6d55ec9090c831169d0945fa4673958bb32841922a176932b0eb581a5fe418dd'}`

## 判断

Kaggle ref `53532418` として提出。public LB `6007.78` でCOMPLETEし、期待値 `6007.780086705443` と一致。current public bestへ採用。

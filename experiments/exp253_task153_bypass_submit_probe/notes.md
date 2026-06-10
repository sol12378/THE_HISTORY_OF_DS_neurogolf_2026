# exp253_task153_bypass_submit_probe

## 目的

exp252で見つかったtask153 `Reshape_node7_to_input0` bypassを、current public best exp234 bundle上で再生成し、提出zipを作る。

## 結果

- improved_and_fullarc: `True`
- evaluation: `{'task_id': 153, 'route': 'crop_or_resize', 'template_name': 'greedy1_Reshape_node7_to_input0', 'baseline_cost': 11212, 'candidate_cost': 10947, 'baseline_points': 15.675260087715321, 'candidate_points': 15.699179274895442, 'file_bytes': 5672, 'validation_status': '265_pass_0_fail', 'status': 'improved', 'reason': 'ok', 'sha256': '53c7fc47a0ee159835b0a7665d93be66fac049a1636353970aac412039790c2e', 'full_arc_ok': True, 'full_arc_passed': 265, 'full_arc_failed': 0, 'full_arc_reason': 'ok'}`
- zip_sanity: `{'count': 400, 'names_ok': True, 'first': ['task001.onnx', 'task002.onnx', 'task003.onnx'], 'last': ['task398.onnx', 'task399.onnx', 'task400.onnx'], 'bytes': 732752, 'sha256': '003d03f3aa73f462863f34f36c55dde1db4cbfe69db3d7b7086bc84340fe4621'}`
- expected_public_lb_if_calibrated: `6006.343919187179`

## 判断

`submit_ready`ならKaggleへ提出してmicro deltaを較正する。

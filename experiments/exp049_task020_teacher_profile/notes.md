# exp049_task020_teacher_profile

## 目的

exp048のP0最大task020について、submit-safe strict artifactとhigh-risk teacher artifactを比較profileする。

## 結果

- strict cost: 90133
- teacher cost: 3931
- teacher point gain: 3.132393
- strict op counts: {'Cast': 67, 'Gather': 71, 'Sum': 7, 'ReduceMax': 2, 'Reshape': 7, 'ArgMax': 3, 'Sub': 12, 'Add': 24, 'Less': 4, 'And': 8, 'Clip': 8, 'Mul': 5, 'Greater': 10, 'Where': 5, 'ReduceSum': 5, 'Equal': 5, 'Concat': 1, 'Pad': 1}
- teacher op counts: {'GatherND': 1, 'Sub': 1, 'Mul': 1, 'ReduceSum': 1, 'Less': 1, 'Cast': 2, 'Unsqueeze': 1, 'MatMul': 2, 'Reshape': 1, 'Squeeze': 1, 'ScatterND': 1}
- teacher largest initializers: [{'label': 'teacher_exp023', 'name': 'safe_name_3', 'dims': '24x24', 'elem_count': 576, 'data_type': 1}, {'label': 'teacher_exp023', 'name': 'safe_name_1', 'dims': '24x11', 'elem_count': 264, 'data_type': 1}, {'label': 'teacher_exp023', 'name': 'safe_name_4', 'dims': '24x6', 'elem_count': 144, 'data_type': 1}, {'label': 'teacher_exp023', 'name': 'safe_name_0', 'dims': '11x4', 'elem_count': 44, 'data_type': 7}, {'label': 'teacher_exp023', 'name': 'safe_name_5', 'dims': '2', 'elem_count': 2, 'data_type': 7}]

## 解釈

teacherは安いが `signature_scatternd_lookup` 由来なので、そのまま提出戦略には使わない。task020の明示ruleを掘るためのteacher/oracleとして使う。

## 次

task020の入出力例を可視/特徴化し、changed-cell maskやobject completion ruleを探索する。

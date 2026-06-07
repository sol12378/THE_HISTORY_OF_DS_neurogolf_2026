# exp045_task031_artifact_profile

## 目的

exp044でfull-grid `GatherND` bbox loweringが破綻したため、task031の既存artifactがcost 18616 に収まっている構造を読む。

## 結果

- status: diagnostic_complete
- manifest cost: 18616
- profiled cost: 18616
- node count: 26
- initializer count: 13
- op counts: {'Slice': 1, 'ReduceSum': 2, 'Greater': 2, 'Cast': 2, 'ArgMax': 4, 'Where': 3, 'Sub': 2, 'Add': 2, 'Gather': 2, 'LessOrEqual': 2, 'Unsqueeze': 2, 'And': 1, 'Pad': 1}
- largest initializers: [{'name': 'safe_name_7', 'dims': '1x12', 'elem_count': 12, 'data_type': 1}, {'name': 'safe_name_12', 'dims': '8', 'elem_count': 8, 'data_type': 7}, {'name': 'safe_name_9', 'dims': '8', 'elem_count': 8, 'data_type': 7}, {'name': 'safe_name_8', 'dims': '7', 'elem_count': 7, 'data_type': 7}, {'name': 'safe_name_0', 'dims': '3', 'elem_count': 3, 'data_type': 7}]

## 解釈

既存artifactのop構成とinitializerサイズを、次のbbox/object compiler設計の制約として使う。特に `GatherND/ScatterND/ArgMax/Where/Conv` の有無と大きなinitializerの有無を見る。

## 次

`node_inventory.csv` と `trace_node_shapes.csv` から、bbox cropがstatic lookup寄りなのか、小さいmask演算寄りなのかを判断する。

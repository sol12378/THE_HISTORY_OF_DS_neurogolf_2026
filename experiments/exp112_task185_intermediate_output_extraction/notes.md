# exp112_task185_intermediate_output_extraction

## Hypothesis

task185の既存artifact内部に、最終出力と同じshapeのone-hot tensorが早い段階で存在するなら、そのtensorを `Identity -> output` にして下流subgraphを丸ごとpruneできる。

## Result

- candidate tensors: `0`
- all intermediate tensors: `241`
- top shape counts: `[{'shape': '', 'count': 135}, {'shape': '30', 'count': 40}, {'shape': '1', 'count': 26}, {'shape': '30x30', 'count': 25}, {'shape': '4', 'count': 6}, {'shape': '9', 'count': 2}, {'shape': '10x3x3', 'count': 1}, {'shape': '1x10x3x3', 'count': 1}, {'shape': '1x30x30', 'count': 1}, {'shape': '3x3x10', 'count': 1}, {'shape': '4x30', 'count': 1}, {'shape': '4x4', 'count': 1}, {'shape': '9x10', 'count': 1}]`
- status counts: `{}`
- validation counts: `{}`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.936157`

## Interpretation

中間output抽出が通れば、局所bypassより大きいsubgraph extractionとして採用する。通らない場合、task185 artifactは最終段まで必要な変換を分散しており、既存graphのsuffix cutではなくfresh fused loweringが必要。

## Risk

- leakage risk: low。既存graphの中間tensorを使うだけで、output lookupは使わない。
- overfitting risk: low-to-medium。acceptedが出た場合はLB calibration候補。

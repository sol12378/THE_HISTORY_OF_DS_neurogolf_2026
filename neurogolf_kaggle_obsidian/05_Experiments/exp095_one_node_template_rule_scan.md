# exp095_one_node_template_rule_scan

## 目的

`exp094` で公式one-hotの1ノード幾何/色channel templateが極小costと分かったため、全400 taskへ `identity/flip/rot180/transpose/channel recolor` 系をPython grid ruleとしてscanする。

## 結果

- candidate rows: `2812`
- full hits: `19`
- geometry hits: `7`
- geometry_recolor hits: `12`

上位には `task150 flip_lr`, `task155 flip_ud`, `task087/task140 rot180`, `task380 rot90` などが含まれた。

## 解釈

Python grid上では単純幾何に見えるtaskがある。ただし公式ONNXでは30x30 padded one-hot tensorを直接変換するため、padding領域の扱いが別問題になる。

## Decision

hitをそのまま採用せず、公式ONNX replacementでfull validationする。

## Risk

- leakage risk: low。
- overfitting risk: pure geometryはlow。color mapはtask-specific。

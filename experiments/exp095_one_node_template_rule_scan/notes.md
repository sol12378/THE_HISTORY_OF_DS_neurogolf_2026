# exp095_one_node_template_rule_scan

## 目的

`exp094` で公式one-hotの1ノード幾何/色channel templateが極小costと分かったため、全400 taskへ `identity/flip/rot180/transpose/channel recolor` 系をPython ruleとしてscanする。

## 結果

- candidate rows: `2812`
- full hits: `19`

## Decision

full hitがあり、baseline costより十分低ければ公式one-hot ONNX replacementを生成してfull validationする。

## Risk

- leakage risk: low。
- overfitting risk: pure geometryはlow。color mapはtask-specificなので、全arc-gen passでもLB較正対象。

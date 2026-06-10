# exp215_task346_count_component_cost_probe

## 目的

task346 ruleのONNX化に必要なcolor countとcomponent-size proxyのcost floorを測る。

## 結果

- baseline_cost: `9178`
- rows: see `result.json`

## 判断

component proxyがbaseline/gainに対して重いなら、task346はsolved-rule assetとして保持し、直接loweringは保留する。

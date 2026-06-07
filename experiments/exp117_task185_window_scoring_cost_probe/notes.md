# exp117_task185_window_scoring_cost_probe

## Hypothesis

task185 window selectorをONNX化する前に、line detection、window scoring、spacing branch coreのcost floorを測る。

## Result

- rows: see `result.json` / `cost_probe.csv`
- local delta: `0.000000`

## Interpretation

selectorが高costなら、full dynamic ONNXではなく、shape/spacing特化または別高gain taskへpivotする。

## Risk

- leakage risk: low。
- overfitting risk: medium。static position proxyは診断用。

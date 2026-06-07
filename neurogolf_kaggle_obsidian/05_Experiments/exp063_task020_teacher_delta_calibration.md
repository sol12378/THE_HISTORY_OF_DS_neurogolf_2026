# exp063_task020_teacher_delta_calibration

## Hypothesis

local/LB較正を早めるため、task020だけteacher artifactへ差し替えたsingle-task deltaを提出できる。

## Result

- strict seed: `exp005_top_cost_rewrite_strict`
- teacher: `exp023_graph_surgery_exp016`
- target task: `20`
- strict cost: `90133`
- teacher cost: `3931`
- theoretical local delta: `+3.132393`
- validation: `24_pass_1_fail`
- reason: `mismatch example 24`

## Decision

不採用。Kaggle提出なし。

## Interpretation

teacher artifactはtask020単体でも利用可能arc-genに対して崩れるため、local/LB較正用にもそのまま使えない。次はexp062の明示ruleをONNX loweringする。

## Risk

- leakage risk: high
- overfitting risk: high

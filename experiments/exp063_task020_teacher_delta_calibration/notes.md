# exp063_task020_teacher_delta_calibration

## 目的

task020だけをteacher artifactへ差し替え、local/LB較正を早める。これは最終戦略ではなく、single-task calibration probe。

## 結果

- validation: 24_pass_1_fail, ok=False
- strict cost: 90133
- teacher cost: 3931
- local delta: 0.000000
- new local estimate: 6282.230228

## Risk

- leakage risk: high。teacher artifactはsignature/public artifact系。
- overfitting risk: high。Kaggle LB deltaでlocalとの対応を見るためだけに使う。

## Decision

validation passならKaggleへ提出し、LB deltaを記録する。

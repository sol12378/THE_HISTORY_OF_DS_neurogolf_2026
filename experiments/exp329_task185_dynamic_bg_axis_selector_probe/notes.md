# exp329_task185_dynamic_bg_axis_selector_probe

## 目的

exp328 の dynamic bg detector を task185 dilated axis selector に接続し、Python selector と全例一致するか、official cost が baseline 未満かを確認する。

## 結果

- selector validation: `267_pass_0_fail`
- selector cost: `41519`
- memory/params: `41428` / `91`
- static: `ok`

## 判断

Selector is ready to connect to compact extraction/core.

## Submission

`no_submit: selector subgraph probe only`

## Risk

- leakage risk: low: input-only bg and geometry selector; no output lookup or public feedback.
- overfitting risk: medium-low: selector rule is task-specific but checked across all local examples.

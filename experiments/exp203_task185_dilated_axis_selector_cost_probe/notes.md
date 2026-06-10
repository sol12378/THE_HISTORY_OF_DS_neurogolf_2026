# exp203_task185_dilated_axis_selector_cost_probe

## 目的

task185 selector proxyをspacing 3/4/5のdilated 4-line windowへ近づけても、selected lattice extractionと合わせてbaseline内に収まるか測る。

## 結果

- best_scored: `dilated_axis_scores_only`
- best_cost: `5208`
- extraction_core_cost_from_exp202: `7660`
- projected_with_extraction: `12868`
- baseline_cost: `59584`

## 判断

projected_under_baselineなら、次はcorrectness-first dynamic-index ONNXへ進む。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。spacing 3/4/5はtask185構造に由来するが、candidateは未生成。

# exp203_task185_dilated_axis_selector_cost_probe

## 目的

task185 selector proxyをspacing 3/4/5のdilated 4-line windowへ近づけても、selected lattice extractionと合わせてbaseline内に収まるか測る。

## 結果

- baseline_cost: `59584`
- `dilated_axis_scores_only`: cost `5208`
- `dilated_axis_selector_argmax`: cost `5236`
- exp202 extraction/core: cost `7660`
- projected_with_extraction: `12868`
- projected_under_baseline: `true`

## 判断

現実寄りselectorでもbaseline内に十分収まる。次はdynamic index生成とbg/special-cell handlingを含むcorrectness-first ONNX candidateへ進む。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。spacing 3/4/5はtask185構造に由来するが、提出candidateではない。

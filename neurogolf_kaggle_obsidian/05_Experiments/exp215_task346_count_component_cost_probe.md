# exp215_task346_count_component_cost_probe

## 目的

task346 ruleのONNX化に必要なcolor countとcomponent-size proxyのcost floorを測る。

## 結果

- baseline_cost: `9178`
- onecell_output_floor: cost `59`
- color_counts_argmin_proxy: cost `111`
- component_growth_1: cost `117135`
- component_growth_4: cost `468135`
- component_growth_8: cost `936135`

## 判断

color count部分は安いが、component-size補正は標準full-grid proxyでは重すぎる。task346はsolved-rule assetとして保持し、直接ONNX loweringは保留する。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: low。candidate未生成。

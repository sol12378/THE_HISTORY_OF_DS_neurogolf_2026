# exp202_task185_selected_lattice_extract_cost_probe

## 目的

task185でselected row/col indexが得られた後、4x4 lattice extraction + homogeneous 2x2 coreがbaseline cost内に収まるか測る。

## 結果

- baseline_cost: `59584`
- `gatherelements_lattice_core_pad`: cost `7660`
- `gathernd_16_cells_core_pad`: cost `38832`
- exp201 axis selector proxy: cost `4156`
- projected_best_with_axis_selector: `11816`
- projected_under_baseline: `true`

## 判断

`GatherElements` 2段 extraction が有望。axis selector proxyと足してもbaselineより十分安いため、task185はdynamic-index correctness candidateへ進める価値がある。

## リスク

- leakage risk: low。cost proxyのみでoutput lookupなし。
- overfitting risk: medium-low。static indexはdynamic selected indexの代用であり、提出candidateではない。

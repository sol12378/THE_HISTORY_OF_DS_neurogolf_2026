# exp202_task185_selected_lattice_extract_cost_probe

## 目的

task185でselected row/col indexが得られた後、4x4 lattice extraction + homogeneous 2x2 coreがbaseline cost内に収まるか測る。

## 結果

- best_scored: `gatherelements_lattice_core_pad`
- best_cost: `7660`
- axis_selector_proxy_cost_from_exp201: `4156`
- projected_best_with_axis_selector: `11816`
- baseline_cost: `59584`

## 判断

projected costがbaseline未満なら、次はdynamic selected indexを実際に作るcorrectness-first loweringへ進む。window pair全列挙は避ける。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。static indexはdynamic selectorの代用で、提出candidateではない。

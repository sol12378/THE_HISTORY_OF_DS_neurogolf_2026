# exp201_task185_axis_selector_cost_probe

## 目的

exp200で成立したtask185 axis-separable selectorについて、pairwise window scoring proxyより安いONNX cost floorになるか測る。

## 結果

- fused axis Conv proxy: cost `4156`
- static Slice+ReduceSum proxy: cost `106064`
- exp117 pairwise static proxy: cost `76194`
- task185 baseline cost: `59584`

## 判断

fused axis scoringならselector部分は十分安い。static Slice方式は中間memoryが重く不採用。次はfused axis scoring + cheap spacing/core pathでcorrectness-first task185 loweringを試す。

## リスク

- leakage risk: low。cost proxyのみ。
- overfitting risk: medium-low。まだ提出candidateではない。

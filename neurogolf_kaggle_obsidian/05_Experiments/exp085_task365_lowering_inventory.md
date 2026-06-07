# exp085_task365_lowering_inventory

## 目的

`task365` の `max_count2 object crop` ruleを600級ONNXへ落とせるか、矩形性とbranch数を棚卸しする。

## 結果

- examples: `266`
- object count: `2` or `3`
- dense rectangle examples: `266/266`
- max objects: `3`
- selected shape count: `16`
- selected shapes: `3x3`〜`6x6`
- max_count2 values: mostly `3` or `4`

## 判断

task365はtask366よりかなりloweringしやすい。dynamic component extractionではなく、dense rectangle window selectorとして低cost化を狙う。

## Next

rectangle-window selector ONNX cost proxyを作る。16 shape branchをそのまま列挙する前に、既存低cost `Slice/Gather/Pad` patternへ収まるか測る。

## Risk

- leakage risk: low。structural inventoryのみ。
- overfitting risk: medium。shape branchの列挙は圧縮が必要。

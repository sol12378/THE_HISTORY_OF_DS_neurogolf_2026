# exp085_task365_lowering_inventory

## 目的

`task365` の `max_count2 object crop` ruleを600級ONNXへ落とせるか、矩形性とbranch数を棚卸しする。

## 結果

- examples: `266`
- object_count_hist: `{3: 185, 2: 81}`
- selected_shape_count: `16`
- dense_rectangle_examples: `266/266`
- max_objects: `3`

## 判断

この実験はinventoryのみでONNXは生成していない。
全objectがdense rectangleなら、dynamic componentではなくrectangle-window selectorとしてloweringを試す。

## Risk

- leakage risk: low。raw lookupなし。
- overfitting risk: medium。shape branchを説明可能に圧縮する必要あり。

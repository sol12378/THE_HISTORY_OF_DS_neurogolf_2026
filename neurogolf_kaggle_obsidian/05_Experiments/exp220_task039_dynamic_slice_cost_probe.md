# exp220_task039_dynamic_slice_cost_probe

## 目的

task039 bbox 3x3 cropを `GatherElements` ではなく dynamic `Slice` で軽量化できるか試す。

## 結果

- status: `rejected`
- reason: `dynamic shape crop`

## 判断

dynamic `Slice` は現行static checkerを通らない。task039はstatic-shape-preservingな別loweringが必要。

## リスク

- leakage risk: low
- overfitting risk: medium-low

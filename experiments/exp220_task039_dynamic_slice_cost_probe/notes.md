# exp220_task039_dynamic_slice_cost_probe

## 目的

task039 bbox 3x3 cropを `GatherElements` ではなく dynamic `Slice` で軽量化できるか試す。

## 結果

- validation: `not_run`
- status: `rejected`
- candidate_cost: ``
- reason: `dynamic shape crop`

## 判断

static shape rejectならdynamic Slice laneは現状保留。通って改善ならbundle化へ進む。

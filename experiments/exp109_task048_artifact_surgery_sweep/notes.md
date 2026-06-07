# exp109_task048_artifact_surgery_sweep

## Hypothesis

task048 is solved by a bridge connectivity rule, but naive 8x8 connectivity lowering is more expensive than baseline. The current artifact may already contain the right structure with removable generated guards or casts. A focused bypass surgery sweep may produce a safe micro-delta.

## Result

- baseline cost: `5609`
- inventory: `{'node_count': 52, 'initializer_count': 15, 'file_bytes': 4683, 'op_counts': {'ArgMax': 1, 'Cast': 4, 'Conv': 12, 'Gather': 1, 'Greater': 13, 'Mul': 1, 'OneHot': 1, 'Pad': 1, 'ReduceSum': 1, 'Reshape': 2, 'Slice': 2, 'Sum': 1, 'Where': 12}, 'duplicate_initializer_group_count': 0}`
- generated candidates: `68`
- status counts: `{'rejected': 67, 'improved': 1}`
- accepted tasks: `[48]`
- local delta: `0.023085`
- new local estimate: `6282.835303`

## Interpretation

If no accepted candidate appears, task048 needs a closed-form bridge feature or a larger subgraph extraction rather than local node bypass.

## Risk

- leakage risk: low。
- overfitting risk: low-to-medium。

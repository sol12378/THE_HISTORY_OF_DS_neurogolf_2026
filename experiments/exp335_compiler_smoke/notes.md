# exp335_compiler_smoke

## Hypothesis
Prove IR -> ONNX -> static -> cost works as one registered runner before trusting task-specific emitters.

## Result
- onnx toolchain available: `True`
- IR cost band: `high_cost_probe_only` / proxy `36010`
- IR hard_reject: `False`
- static gate: `True` (ok)
- candidate sha256: `2fe86edd1a36ac8e9704e83cc5237b964d83bb22e353d79445a4e479e2bd5af8`
- onnx cost band: `high_cost_probe_only`

## Decision
No submit, no bundle. This is the compiler smoke bridge gate; adoption is false by construction.

## Next
Install onnx/onnxruntime, then add task-baseline compiler probes (task251_mask_cost_probe, task037_sparse_writeback_probe).


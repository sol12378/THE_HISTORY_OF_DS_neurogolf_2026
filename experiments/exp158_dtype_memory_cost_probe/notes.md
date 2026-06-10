# exp158_dtype_memory_cost_probe

## Hypothesis

full-grid中間をfp16/bool/uint8へ縮小できれば、公式memory costがdtype幅に比例して下がる可能性がある。

## Result

- status: `cost_probe_complete`
- decision: dtype_memory_reduction_observed; prioritize targeted post-pass probes: fp16:36002->18000, bool:36002->9000, uint8:36002->9000

| case | memory | params | cost | reason |
|---|---:|---:|---:|---|
| identity | 0 | 0 | 0 | ok |
| fp32_add_zero | 0 | 1 | 1 | ok |
| fp32_add_zero_then_identity | 36000 | 1 | 36001 | ok |
| fp32_mul_one_then_add_zero | 36000 | 2 | 36002 | ok |
| fp16_roundtrip | 18000 | 0 | 18000 | ok |
| bool_roundtrip | 9000 | 0 | 9000 | ok |
| uint8_roundtrip | 9000 | 0 | 9000 | ok |
| int32_roundtrip | 36000 | 0 | 36000 | ok |
| fp16_add_zero_roundtrip | 36000 | 1 | 36001 | ok |
| bool_where_roundtrip | 9000 | 9000 | 18000 | ok |

## Leakage / Overfitting Risk

low: synthetic cost-only probe; no task labels used.
low: measures score_network cost physics, not public LB behavior.

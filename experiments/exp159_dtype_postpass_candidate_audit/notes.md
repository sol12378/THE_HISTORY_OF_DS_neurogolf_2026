# exp159_dtype_postpass_candidate_audit

## Hypothesis

exp152 current best bundleには、bool/mask系full-grid中間をFLOATとしてmaterializeしているtaskがあり、dtype post-pass候補を構造監査で絞れる。

## Result

- status: `candidate_audit_complete`
- audited_tasks: `400`
- decision: Inspect top candidates for local graph rewrites that keep boolean/mask intermediates as BOOL/UINT8 and avoid fp32 materialization.

## Top Candidates

| task | heuristic_saving | full_grid_float | boolish_full_grid | cast_boolish_to_float | where_full_grid | ops |
|---:|---:|---:|---:|---:|---:|---|
| 366 | 4275000 | 3 | 129 | 2 | 82 | Where:145 And:111 Equal:77 Cast:61 ReduceSum:44 Greater:25 Tile:25 Or:22 MaxPool:22 Reshape:21 Gather:18 Sub:16 |
| 284 | 1899000 | 4 | 69 | 0 | 4 | And:27 Equal:13 Add:11 GreaterOrEqual:10 LessOrEqual:10 Or:10 Where:8 Sub:7 Reshape:7 ArgMax:5 ReduceSum:3 Gather:3 |
| 158 | 1494000 | 21 | 37 | 18 | 1 | Slice:91 Reshape:91 And:72 Greater:55 Less:37 Cast:19 MatMul:2 Flatten:1 Unsqueeze:1 Sub:1 Mul:1 ReduceSum:1 |
| 382 | 1449000 | 1 | 51 | 0 | 8 | Slice:45 Pad:36 Where:30 And:29 Equal:28 Greater:23 Or:20 ReduceMax:8 ReduceMin:4 Sub:4 CumSum:4 Cast:3 |
| 187 | 1215000 | 1 | 45 | 0 | 0 | Greater:24 And:16 Cast:15 MaxPool:14 Gather:10 ReduceSum:1 LessOrEqual:1 Or:1 Not:1 Concat:1 |
| 328 | 1098000 | 4 | 37 | 1 | 8 | And:21 Or:12 Cast:10 Greater:10 Where:8 Less:5 Equal:4 GatherND:4 Not:4 Sub:3 Concat:3 Unsqueeze:3 |
| 034 | 855000 | 1 | 30 | 0 | 5 | And:21 Less:12 ReduceSum:8 Where:7 Sum:5 Sub:5 Equal:4 Or:4 Cast:3 Gather:2 Abs:2 Conv:1 |
| 054 | 603000 | 2 | 16 | 0 | 19 | Where:23 Gather:14 Mul:14 Greater:9 Max:9 Equal:7 Sub:6 Conv:5 Cast:4 ReduceMax:4 CumSum:4 MaxPool:4 |
| 383 | 576000 | 1 | 21 | 0 | 1 | Greater:9 And:9 Cast:6 ReduceSum:6 Conv:5 Or:5 Where:4 Mul:3 Slice:2 Sub:2 Not:2 Equal:2 |
| 377 | 531000 | 9 | 18 | 1 | 2 | Cast:28 Reshape:23 Less:22 Where:20 Not:20 ReduceMax:18 And:17 Min:13 ReduceMin:10 Concat:6 GatherND:5 Equal:5 |
| 138 | 486000 | 2 | 17 | 0 | 3 | Cast:14 Squeeze:10 Gather:10 And:9 Greater:7 Equal:7 Where:7 ReduceMax:5 Or:5 ReduceSum:4 Less:4 Add:4 |
| 182 | 405000 | 1 | 15 | 0 | 0 | Equal:60 And:45 Cast:14 Or:12 Slice:11 Greater:6 ReduceSum:6 ReduceMax:4 Conv:2 ConvTranspose:2 Concat:2 Clip:1 |

## Leakage / Overfitting Risk

low: graph structure audit only.
low-to-medium: cost-structure heuristic may not translate to valid rewrites.

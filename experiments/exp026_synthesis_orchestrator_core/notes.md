# exp026_synthesis_orchestrator_core

## Hypothesis

7600を狙うには、公開artifact blendの足し算ではなく、taskごとに説明可能なDSL programを合成し、公式costに近い制約でONNXへ落とすオーケストレータが必要である。

## Result

- base: `experiments\exp023_graph_surgery_exp016`
- base local estimate: `6479.398825`
- gap to 6500: `20.601175`
- gap to 7600: `1120.601175`
- queue items: `400`

## Phase Projection

- `phase1_6500_bridge`: top100をcost<=9000にできれば `+133.562`
- `phase2_7000_submit_safe`: top200をcost<=3000にできれば `+396.073`
- `phase3_7400_private_like`: top320をcost<=1000にできれば `+829.305`
- `phase4_7600_contender`: top400をcost<=300にできれば `+1284.967`

## Top Queue

| rank | task | family | current cost | target | projected gain | first action |
|---:|---:|---|---:|---:|---:|---|
| 1 | 286 | `signature_lookup_current` | 110902 | 9000 | 2.511 | mine train/test signatures for invariant features |
| 2 | 187 | `point_to_line_pattern` | 105313 | 9000 | 2.460 | identify seed colors and directions |
| 3 | 198 | `line_grid_fill` | 103821 | 9000 | 2.445 | detect complete/incomplete rows and columns |
| 4 | 203 | `region_partition_fill` | 88402 | 9000 | 2.285 | detect boundary colors and line-grid rooms |
| 5 | 398 | `crop_resize` | 84295 | 9000 | 2.237 | enumerate fixed crop, object-centered crop, mask-color crop |
| 6 | 313 | `region_partition_fill` | 83572 | 9000 | 2.228 | detect boundary colors and line-grid rooms |
| 7 | 255 | `same_shape_global_transform` | 74403 | 9000 | 2.112 | test flip/rotate/transpose/color-map sketches |
| 8 | 29 | `signature_lookup_current` | 71139 | 9000 | 2.067 | mine train/test signatures for invariant features |
| 9 | 107 | `crop_resize` | 73131 | 9000 | 2.095 | enumerate fixed crop, object-centered crop, mask-color crop |
| 10 | 280 | `signature_lookup_current` | 70515 | 9000 | 2.059 | mine train/test signatures for invariant features |
| 11 | 137 | `point_to_line_pattern` | 72472 | 9000 | 2.086 | identify seed colors and directions |
| 12 | 145 | `line_grid_fill` | 69757 | 9000 | 2.048 | detect complete/incomplete rows and columns |

## Worker PDCA

- main agentはorchestratorとして、`worker_task_queue.csv` の上位から低reasoning workerに狭い候補生成を渡す。
- worker成果物は必ずmainがreviewし、基準未満なら同じtask/familyで再帰的に修正させる。
- 採用条件は `validation pass`, `candidate cost < baseline`, `static/banned-op gate pass`, `proof logあり`。

## Risks

- leakage risk: high。現best baseはexp023で、exp016由来のsignature lookup local upper boundを含む。
- overfitting risk: high。Phase 3以降はfamily holdoutとfull arc-gen validationが必要。

## Next PDCA

1. `exp027_cost_aware_lowering_bench` で過去失敗loweringをguardrail化する。
2. `exp028_lookup_to_rule_miner` でsignature lookup 196件をrule候補へ圧縮する。
3. `exp029_crop_object_synthesizer` でcrop/resize 85件をSlice/Gather中心に置換する。

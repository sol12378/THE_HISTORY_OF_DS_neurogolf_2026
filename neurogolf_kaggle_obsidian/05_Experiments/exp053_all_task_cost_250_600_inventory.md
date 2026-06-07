# exp053_all_task_cost_250_600_inventory

## Hypothesis

大量のtaskをcost 250〜600台へ落とすには、まず全400 taskについて現在cost、600/250目標との差、family、禁止lowering、次actionを一つのmaster queueにする必要がある。

## Result

- source: `experiments/exp048_submit_safe_seed_inventory/task_inventory.csv`
- backlog: `experiments/exp_b001_rule_replacement_backlog/rule_replacement_backlog.csv`
- strict seed score: `6282.230228`
- cost<=600 already: `27/400`
- cost<=250 already: `19/400`
- need cost reduction to <=600: `373/400`
- projected score if all tasks are at least cost<=600: `7503.630200`
- projected score if all tasks are at least cost<=250: `7833.830824`
- margin over 7700 at <=600 floor: `-196.369800`
- margin over 7700 at <=250 floor: `+133.830824`

## Family Gap

| Family | Tasks | Need >600 | Gain to 600 | Gain to 250 |
|---|---:|---:|---:|---:|
| signature_lookup_current | 196 | 196 | 834.211 | 1005.803 |
| crop_resize | 85 | 73 | 150.535 | 216.206 |
| sparse_edit_or_object_completion | 57 | 49 | 95.414 | 139.461 |
| point_to_line_pattern | 28 | 27 | 69.981 | 93.619 |
| region_partition_fill | 15 | 12 | 33.197 | 44.443 |
| line_grid_fill | 14 | 14 | 32.504 | 44.760 |

## Interpretation

cost<=600だけでは7700に届かない。したがって「600台へ落とす」は中間目標であり、上位familyは250寄りの最小回路を狙う必要がある。

最大の差分は `signature_lookup_current`。ただしteacher artifactをそのまま使うとexp041のようにLB崩壊するため、明示rule / DSL / DAGへ圧縮してから低cost ONNXへloweringする。

## Outputs

- `experiments/exp053_all_task_cost_250_600_inventory/task_cost_targets.csv`
- `experiments/exp053_all_task_cost_250_600_inventory/family_cost_gap.csv`
- `experiments/exp053_all_task_cost_250_600_inventory/cost_bucket_summary.csv`
- `experiments/exp053_all_task_cost_250_600_inventory/result.json`

## Decision

`task_cost_targets.csv` を全task 250〜600化のmaster queueにする。次は上位gap familyに対して、submit-safeなrule searcherとcost-aware loweringを実装する。

## Risk

- leakage risk: medium。teacher artifactはoracleであり提出候補ではない。
- overfitting risk: medium-high。全arc-gen一致だけではhidden汎化の証明にならないため、小さいcalibration bundleが必要。

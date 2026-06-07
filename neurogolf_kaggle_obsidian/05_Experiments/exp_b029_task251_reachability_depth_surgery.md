# exp_b029_task251_reachability_depth_surgery

## 目的

task251既存artifactのreachability depthを削れるか確認する。`exp_b028` でnaive flood-fill unrollが高costだったため、既存artifact内の到達計算の後段を短縮できれば低cost化できる可能性がある。

## 結果

- base: `exp_b025_submit_safe_delta_union`
- task: `251`
- candidates: `7`
- valid depth reductions: `0`

| replacement | validation |
|---|---:|
| `safe_name_58` | `259_pass_1_fail` |
| `safe_name_55` | `88_pass_1_fail` |
| `safe_name_52` | `4_pass_1_fail` |
| `safe_name_49` | `4_pass_1_fail` |
| `safe_name_46` | `2_pass_1_fail` |
| `safe_name_43` | `1_pass_1_fail` |
| `safe_name_40` | `0_pass_1_fail` |

## 解釈

既存artifactのreachability depthはarc-gen上ほぼ必要。単純に後段を切るgraph surgeryでは低cost化できない。

## Decision

task251はruleとしては解けているが、既存artifact depth pruningもnaive unrollも不採用。次はclosed-form / rectangle-specific mask、または別task laneへ移る。

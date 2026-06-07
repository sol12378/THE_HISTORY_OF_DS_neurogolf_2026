# exp_b028_task251_closed_component_lowering_probe

## 目的

`exp_b027` の task251 closed zero-component rule を、既存 `boundary_flood_fill` ONNX loweringで実cost評価する。

## 結果

- base: `exp_b025_submit_safe_delta_union`
- task: `251`
- base cost: `100580`
- step 6: validation `97_pass_1_fail`
- step 8: validation `266_pass_0_fail`, cost `138631`
- step 10: validation `266_pass_0_fail`, cost `158431`
- step 12: validation `266_pass_0_fail`, cost `178231`
- step 18: validation `266_pass_0_fail`, cost `237631`
- best accepted improvement: none
- submission: none

## 解釈

task251 rule自体は正しいが、naive boundary flood-fill unrollはbaselineより高い。これはexp019/task187と同じ失敗パターンで、region fill系は正解ruleを見つけるだけでは足りない。

## Decision

`boundary_flood_fill` unrollはtask251にも使わない。次はrectangle-specific closed mask、border reachabilityのより短い表現、または既存artifact surgeryを試す。

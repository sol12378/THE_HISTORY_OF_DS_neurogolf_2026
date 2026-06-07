# exp_b028_task251_closed_component_lowering_probe

## 目的

`exp_b027` の task251 closed zero-component rule を、既存 `boundary_flood_fill` ONNX loweringで実cost評価する。

## 結果

- base cost: `100580`
- best candidate: `none`
- local delta: `0.000000000`
- submission decision: `no_submit`

## 判断

改善があればsingle-task deltaとして提出する。改善がなければ、naive flood-fill系は避け、rectangle-specific closed maskまたはborder reachabilityの低cost化へ進む。

# exp065_task020_lowering_reference

## 目的

task020 exp062 ruleをONNX化する前に、bbox crop / target color / class selector / orientation / writebackのreference実装を固定する。

## 結果

- pass: 266/266
- split pass: {'train': 3, 'test': 1, 'arc-gen': 262} / {'train': 3, 'test': 1, 'arc-gen': 262}
- reasons: {'ok': 266}
- class counts: {'corners': 87, 'inner_diag': 87, 'edge_mid': 92}

## Decision

referenceがfull passならONNX化へ進む。partialならtarget-color selectionやclass selectorを修正する。

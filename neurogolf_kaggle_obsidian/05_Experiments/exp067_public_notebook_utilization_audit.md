# exp067_public_notebook_utilization_audit

## 目的

ユーザー提示の公開notebook群を、そのままblendへ再投入するのではなく、strict-safeなsurgery queue、compiler lane prior、source provenanceへ変換する。

## 入力

- `massimilianoghiotto/convolution-series-part-3`
- `konbu17/neurogolf-2026-blended-till-4-27`
- `vyankteshdwivedi/neurogolf-multi-source-onnx-solver`
- `seddiktrk/surgical-onnx-precision-parameter-reduction`
- `karnakbaevarthur/all-task-description-analysis`
- `magmacot/neurogolf-new-blending`
- `needless090/neurogolf-4250`

## 結果

- exp002に既に含まれていた系統:
  - `massimilianoghiotto`
  - `konbu17`
  - `vyankteshdwivedi`
  - `magmacot`
  - `needless090`
- 新規取得:
  - Seddik notebook `.ipynb`
  - Karnak notebook `.ipynb`
  - Karnak `neurogolf-2026-task-transformation-library`

## Seddik-style Audit

current base: `exp066_task020_correctness_onnx_lowering`

- scanned: 400 tasks
- tasks with unused initializers: 1
- tasks with duplicate initializers: 8
- tasks with uniform-safe scalarization: 28
- total unused params: 31
- total duplicate wasted params: 55
- total uniform-safe saveable params: 4583

Top surgery queue:

- task383: uniform-safe 899 params, Pattern_Recognition
- task284: uniform-safe 899 params, Pattern_Recognition
- task110: uniform-safe 840 params, Pattern_Recognition
- task117: uniform-safe 462 params, Pattern_Recognition
- task017: uniform-safe 440 params, Pattern_Recognition
- task077: uniform-safe 419 params, Pattern_Recognition

## Karnak Description Library

- rows: 400
- primary categories:
  - Pattern_Recognition: 160
  - Object_Based: 158
  - Spatial_and_Geometric: 47
  - Color_and_Logical: 35
- top transformations:
  - Object Detection: 262
  - Color Mapping: 244
  - Filling Regions: 196
  - Translation: 100
  - Cropping: 87
  - Gravity: 79

## 解釈

公開blend出力は既にexp002でかなり使っている。追加で同じ系統を混ぜるより、Seddik式の外科的圧縮をstrict seed post-passへ入れ、Karnakのtask descriptionをcompiler lane priorとして使う方が有意義。

## Decision

次は `seddik_surgery_audit.csv` 上位taskへfull-arc gated scalarization/dedupを試す。Karnak分類は `exp053` cost target queueとjoinして、L1/L2 sparse fill、L4 shape/crop、region/fill系の探索順序を再重み付けする。

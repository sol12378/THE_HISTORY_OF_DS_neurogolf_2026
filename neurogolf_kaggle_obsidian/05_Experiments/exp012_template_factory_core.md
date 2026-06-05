# exp012_template_factory_core

## 目的

GPU route classifierのrankingを使い、top cost task向けにstatic ONNX templateを自動生成する。

## 実装

- `conv_color_map`
- `fixed_mask_rewrite`
- `constant_output`
- `fixed_slice_crop`
- `fixed_geometry_*`
- `signature_scatter_lookup`

## 結果

- input: `exp005_top_cost_rewrite_strict`
- top_k: `400`
- arc_gen_sample: `20`
- local estimate: `6296.2974`
- baseline: `6282.2302`
- delta: `+14.0671`
- improved tasks: `38`
- adopted template: `signature_scatter_lookup`

## 判断

6500には未達。lookup系はlocal上限探索としては有効だが、arc-gen labelを利用するためleakage/overfitting riskが高く、提出候補にはしない。

## 次PDCA

- top cost taskごとのgraph surgeryを強化する。
- object completion用の小型CNN/conv templateをGPUで探索し、strict validationで採用する。
- public/local寄せではなく、primitive別holdoutでprivate-like riskを測る。

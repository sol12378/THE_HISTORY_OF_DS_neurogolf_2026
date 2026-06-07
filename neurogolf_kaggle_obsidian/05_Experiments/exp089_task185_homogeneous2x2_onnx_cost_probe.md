# exp089_task185_homogeneous2x2_onnx_cost_probe

## 目的

`task185` のcore ruleである4x4 lattice matrix -> 3x3 homogeneous 2x2 block判定が、ONNX cost `<=600` に入るかを測る。

## 結果

- `static_slice_4x4_to_3x3_shape_probe`: shape mismatchで診断のみ。
- `homogeneous2x2_conv_proxy`: cost `1147` (`memory=1090`, `params=57`)

## 解釈

4x4 lattice matrixが既に取れている前提でも、grouped 2x2 Conv + Equal + Cast の同色判定は600を超える。dynamic lattice extractionを追加するとさらに重くなる。

## Decision

task185は正答rule hitとして保持するが、標準Conv equality loweringは600級routeではない。次はConvを使わない同色判定、または既存artifact surgeryを試す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static Sliceはproxyであり、提出候補ではない。

# exp083_sparse_update_primitive_cost_probe

## 目的

`task366` のsmall-patch / sparse-coordinate lowering候補として、少数 `ScatterND` 更新や小さい `Gather` の公式costを測る。

## 結果

| primitive | updates | cost |
|---|---:|---:|
| constantofshape_scatternd | 7 | 36039 |
| input_zero_scatternd | 7 | 36036 |
| small_gather_tile_proxy | 7 | 10418 |
| constantofshape_scatternd | 49 | 36249 |
| input_zero_scatternd | 49 | 36246 |
| small_gather_tile_proxy | 49 | 72104 |

## 判断

small sparse updateでも600級には届かない。`ScatterND` は更新数が少なくても30x30 stateをmaterializeするため高cost。`Gather` も固定output shapeへ戻すと重くなる。

`task366` は `266/266` rule hitとして保持するが、標準的なfull-grid/sparse-writeback ONNX loweringではscore-producingではない。

## Next

- task366は別表現が見つかるまでlowering保留。
- 次のscore-producing PDCAは、既存低cost artifactの `Gather` / static transform / small Conv 型へ近いtaskを狙う。

## Risk

- leakage risk: low。合成primitiveのみ。
- overfitting risk: low。cost-only診断。

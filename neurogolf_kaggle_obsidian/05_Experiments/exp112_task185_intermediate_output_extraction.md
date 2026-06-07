# exp112_task185_intermediate_output_extraction

## Hypothesis

task185の既存artifact内部に、最終出力と同じshapeのone-hot tensorが早い段階で存在するなら、そのtensorを `Identity -> output` にして下流subgraphを丸ごとpruneできる。

## Result

- campaign index: `14`
- baseline cost: `59584`
- target output shape: `1x10x30x30`
- all intermediate tensors: `241`
- output-shape intermediate tensors: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.93615685804`

Top shape counts:

- unknown shape: `135`
- `30`: `40`
- `1`: `26`
- `30x30`: `25`
- `4`: `6`
- `1x10x3x3`: `1`

## Interpretation

task185の既存artifactは、最終 `[1,10,30,30]` one-hot outputを途中で作ってから後段で加工している形ではない。suffix cut / intermediate output extractionでは削れない。

これは重要な否定結果で、task185は既存graphの大きな後段pruneではなく、Python ruleで解けている「4x4 lattice -> homogeneous 2x2 block」のfresh fused loweringが必要。

## Next Action

task185の次手は、出力3x3のcoreだけでなく、paddingを含めた最終 `[1,10,30,30]` を少数nodeで直接生成するfused loweringに切り替える。既存artifact surgeryは優先度を下げる。

## Risk

- leakage risk: low。graph-only extractionの否定実験。
- overfitting risk: low。

# exp114_task185_fresh_fused_lowering_probe

## Hypothesis

task185 fresh fused loweringで、最終official output shapeまで含めたcost floorを測る。`Slice` extraction、direct shifted `Slice+Mul`、sparse `GatherND` extractionを比較し、安いvariantはfull validationも確認する。

## Result

- campaign index: `16`
- baseline cost: `59584`
- best cost proxy: `slice_conv_pad_final`, cost `1516`
- direct four-slice Mul cost: `2569`
- GatherND lattice extraction cost: `38832`
- full validation: all `0_pass_1_fail`
- accepted tasks: `[]`
- local delta: `0.000000`

## Interpretation

homogeneous-2x2 core自体は、最終Pad込みでも `1516` まで下げられる。baseline `59584` から見れば大きい。ただし固定lattice位置は全例に合わず、提出候補ではない。

`GatherND` extractionは30x30 full-grid transposeを作るため非常に重い。task185では sparse coordinate extraction より、grid-line spacing/offsetを推定して小さな `Slice/Conv/Pad` に流す方がよい。

## Next Action

固定座標ではなく、lattice行/列のspacingとoffsetをinputから推定するdynamic grid-line compilerへ進む。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice proxyであり、correctness候補ではない。

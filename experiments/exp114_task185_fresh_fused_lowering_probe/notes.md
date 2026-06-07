# exp114_task185_fresh_fused_lowering_probe

## Hypothesis

task185 fresh fused loweringで、最終official output shapeまで含めたcost floorを測る。`Slice` extraction, direct shifted `Slice+Mul`, sparse `GatherND` extractionを比較する。

## Result

- rows: see `result.json` / `cost_probe.csv`
- best variant: `slice_conv_pad_final`
- best cost: `1516`
- validation counts: `{'0_pass_1_fail': 3}`
- accepted tasks: `[]`
- local delta: `0.000000`

## Interpretation

これはcost probeであり提出候補ではない。600級に届くvariantがあればfull correctness loweringへ進む。届かない場合は、標準ONNX primitiveの組み合わせではtask185の250〜600化が難しく、よりfusedな表現または別taskへ優先度を移す。

## Risk

- leakage risk: low。
- overfitting risk: medium。static lattice位置proxyで、全例correctnessは主張しない。

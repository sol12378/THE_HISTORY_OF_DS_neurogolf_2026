# exp024_ring_depth_dynamic_colormap

## Hypothesis

task203 の同心ring反転は、ring上の色サンプルから動的な `10x10` color map を作り、全ピクセルに適用すれば static ONNX で表せる。

## Result

- target task: `task203`
- baseline cost: `88402`
- candidate: `ring_depth_reverse_color_map`
- validation: `25_pass_0_fail`
- candidate cost: `216708`
- status: `no_gain`

## Interpretation

Python ruleは正しく、ONNXもvalidationを通った。ただし、全30x30ピクセルへ動的mappingを `MatMul` で適用する lowering は memory cost が大きく、baseline artifactに負ける。

次は全ピクセルmappingではなく、ring depthごとの小さなmask、または既存artifactのgraph surgeryで `task203` を狙う。

## Risks

- leakage risk: medium。rule自体は幾何的だが、検証はsample20。
- overfitting risk: medium。size候補を `6..18` の偶数squareに固定している。


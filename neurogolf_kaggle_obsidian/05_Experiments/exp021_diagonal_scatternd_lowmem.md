# exp021_diagonal_scatternd_lowmem

## Hypothesis

task398 style の diagonal shift tile は、full-grid `Where` ではなく sparse `ScatterND` に落とせば、baseline artifact より低costになる可能性がある。

## Result

- base: `exp016_top100_rewrite_campaign`
- target task: `task398`
- local estimate: `6479.383086`
- delta: `0.000000`
- status: `no_gain`
- `diagonal_shift_tile_scatternd` は validation 可能だったが、candidate cost は `369873` で baseline `84295` より重かった。
- `signature_scatternd_lookup` も candidate cost `138254` で baseline より重く、採用されなかった。

## Interpretation

rule を発見できても、それをそのまま大きな座標initializerや広い `ScatterND` に落とすと official-like cost で負ける。7600を狙うには、ONNX生成後にrejectするだけでは遅く、生成前にcostを予測して lowering plan を選ぶ必要がある。

## Risks

- leakage risk: high。base は exp016 の signature lookup を含む。
- overfitting risk: high。task398単体probeであり、full validationではない。


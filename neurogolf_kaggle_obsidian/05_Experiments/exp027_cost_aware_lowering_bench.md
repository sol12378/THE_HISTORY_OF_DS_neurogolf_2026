# exp027_cost_aware_lowering_bench

## 目的

過去のcorrect-but-expensive候補を集約し、ONNX生成前に弾くcost-aware guardrailを作る。

## 結果

- audited rows: `24563`
- validation-pass no-gain rows: `228`
- improved rows: `195`
- base local estimate: `6479.398825`

## 主なguardrail

- full-grid `Where` chainは事前reject候補。
- full-grid `Tile + mask` は事前reject候補。
- dynamic `MatMul` color mapは事前reject候補。
- large coordinate initializer付き `ScatterND` は事前reject候補。
- memorized `signature_scatternd_lookup` はlocal upper boundとしてのみ扱い、7600/PB狙いではrule miningへ戻す。

## 採用しやすいlowering

- constant `Slice`
- small `Gather`
- small `Conv`
- compact row/column mask
- initializer pruning / graph surgery

## 次

候補生成器は `expected_cost_upper_bound` を持ち、上限超過ならONNX emissionしない。

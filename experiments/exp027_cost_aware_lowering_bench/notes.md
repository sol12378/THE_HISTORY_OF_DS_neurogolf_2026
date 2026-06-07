# exp027_cost_aware_lowering_bench

## Hypothesis

7600向けprogram synthesisでは、候補ONNXを大量に生成する前にcost-aware gateを置く必要がある。過去の失敗候補を集約すれば、validation passしても重くなるloweringを事前に止められる。

## Result

- observed candidate rows: `24563`
- validation-pass but no-gain rows: `228`
- known improved rows: `195`
- base local estimate remains: `6479.398825`

## Main Guardrails

- 長い `Where` chain、全画素 `Tile`、動的 `MatMul` color map、大きい座標initializer付き `ScatterND` はONNX生成前に概算cost gateへ通す。
- `Slice`, `Gather`, small `Conv`, initializer pruning は優先して試す。
- `signature_scatternd_lookup` はlocal upper boundとしては使えるが、7600/PB狙いではrule miningへ戻す。

## Worker Audit

低reasoning workerの監査でも、exp019-025の失敗は同じ傾向だった。正しいruleでもfull-grid/dynamic/large-initializer loweringはbaseline artifactに負けるため、main orchestratorがworker成果物をreviewし、cost forecast未達なら再帰的に差し戻す。

## Next PDCA

1. `exp028_lookup_to_rule_miner` でsignature lookupの圧縮候補を作る。
2. `exp029_crop_object_synthesizer` でconstant Slice/Gather系を増やす。
3. loweringごとにexpected_cost_upper_boundを持たせ、上限超過ならONNX emissionしない。

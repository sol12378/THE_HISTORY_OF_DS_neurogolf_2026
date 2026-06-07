# exp022_program_synthesis_pipeline_7600

## Hypothesis

7600を狙うには、個別taskの手作業template追加では足りない。必要なのは、task family分類、DSL候補生成、ONNX lowering、static/rule検証、cost最小化、採用監査を同じ形で回すprogram synthesis pipelineである。

## 現状

- base local estimate: `6479.383086`
- gap to 6500: `20.616914`
- gap to 7600: `1120.616914`
- exp021のtask398 probeでは、ruleはvalidation可能だったが `ScatterND` loweringのcostがbaselineより重く、採用されなかった。

## 必要なシステム

- `lookup compression and rule extraction`: 196 tasks
- `object-anchor crop/resize synthesizer`: 85 tasks
- `sparse edit and object completion enumerator`: 57 tasks
- `seed-to-pattern grammar`: 28 tasks
- `region and room-fill synthesizer`: 15 tasks
- `line, ray, and grid grammar`: 14 tasks
- `color role normalizer`: 4 tasks
- `global transform and artifact surgery`: 1 tasks

## 段階別Acceptance

- Phase 1 / 6500: top cost taskを中心に低cost loweringを作り、local estimate >= 6500。submitは別途full validation後。
- Phase 2 / 7000: high-risk lookupをDSL programへ置換し、提出安全性を上げる。
- Phase 3 / 7400: primitive family別holdoutでprivate耐性を測る。
- Phase 4 / 7600: enumerator、cost model、lowering optimizer、proof logを統合し、改善閾値ごとのsubmit cadenceに乗せる。

## 次の実装優先度

1. `static_lookup_compression`: signature lookup候補から実ルールを抽出し、memorization依存を下げる。
2. `object_anchor_crop`: `NonZero` なしでbbox/cropを表す低cost loweringを作る。
3. `sparse_coordinate_program`: full-grid `Where` を避け、少数座標の `GatherND/ScatterND` に落とす。
4. `region_partition_fill`: naive flood unrollではなく、line-grid/room前提の閉領域判定に寄せる。
5. `cost_model`: ONNX生成前にmemory+paramsを予測し、高cost loweringを生成前に棄却する。

## Risks

- leakage risk: high。現baseはexp016のsignature lookupを含む。
- overfitting risk: high。local/sampleだけでなくfull arc-genとfamily holdoutが必要。

# exp022_program_synthesis_pipeline_7600

## Hypothesis

7600を狙うには、個別taskの手作業template追加では足りない。task family分類、DSL候補生成、ONNX lowering、static/rule validation、cost最小化、採用監査を一体化した program synthesis pipeline が必要である。

## Result

- base local estimate: `6479.383086`
- gap to 6500: `20.616914`
- gap to 7000: `520.616914`
- gap to 7400: `920.616914`
- gap to 7600: `1120.616914`
- backlog: `experiments/exp022_program_synthesis_pipeline_7600/synthesis_backlog.csv`
- result: `experiments/exp022_program_synthesis_pipeline_7600/result.json`

## Family Counts

- `signature_lookup_current`: 196
- `crop_resize`: 85
- `sparse_edit_or_object_completion`: 57
- `point_to_line_pattern`: 28
- `region_partition_fill`: 15
- `line_grid_fill`: 14
- `color_map`: 4
- `same_shape_global_transform`: 1

## Required Systems

- `lookup compression and rule extraction`: signature lookup の暗記候補をDSL ruleへ置換する。
- `object-anchor crop/resize synthesizer`: `NonZero` / `Compress` なしでbbox/crop/padを低costに表す。
- `sparse edit and object completion enumerator`: 少数座標の `GatherND/ScatterND` と小kernelで補完する。
- `seed-to-pattern grammar`: 点やanchorから線、ray、反復patternを生成する。
- `region and room-fill synthesizer`: flood unrollではなく line-grid / room 構造の閉領域判定へ寄せる。
- `line, ray, and grid grammar`: 行列方向、停止条件、色役割を探索する。
- `cost model`: ONNX生成前に official-like memory+params を見積もり、高cost loweringを棄却する。

## Phase Gates

- Phase 1 / 6500: top100 を中心に低cost化し、local estimate >= 6500。submitはfull validationとrisk audit後。
- Phase 2 / 7000: high-risk lookupをDSL programへ置換し、full arc-gen 400/400を通す。
- Phase 3 / 7400: primitive family別holdoutでprivate-like劣化を測る。
- Phase 4 / 7600: enumerator、lowering optimizer、proof log、submission cadenceを統合する。

## Projection

- top100 cost <= 9000: projected local `6612.947773`
- top200 cost <= 3000: projected local `6875.463728`
- top320 cost <= 1000: projected local `7308.703668`
- top400 cost <= 300: projected local `7764.366304`

## Risks

- leakage risk: high。base は exp016 の signature lookup を含む。
- overfitting risk: high。sample-localとarc-gen依存が残るため、family holdoutが必要。


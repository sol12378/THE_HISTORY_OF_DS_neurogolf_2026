# exp018_neurogolf_dsl_core

## Hypothesis

7600を狙うには、公開artifact blendや単発template追加ではなく、タスクをDSL familyへ分解して、
候補生成、検証、ONNX lowering、cost最適化を同じ形式で回す必要がある。

## Result

- base: `experiments\exp016_top100_rewrite_campaign`
- local estimate from base: `6479.383080`
- gap to 6500: `20.616920`
- gap to 7600: `1120.616920`
- primitive count: `7`

## Family Counts

- `color_map`: 4
- `crop_resize`: 85
- `line_grid_fill`: 14
- `point_to_line_pattern`: 28
- `region_partition_fill`: 15
- `same_shape_global_transform`: 1
- `signature_lookup_current`: 196
- `sparse_edit_or_object_completion`: 57

## Interpretation

exp017の単純な反復近傍fillはsample20では改善しなかった。
残り上位cost taskは、領域分割、線/部屋構造、点から線・パターン生成、object anchor cropの比率が高い。
次PDCAは `boundary_flood_fill` と `rectangular_room_fill` をONNX loweringまで実装し、task187/198/137/203/286周辺を直接狙う。

## Risks

- leakage risk: high。現baseはexp016で、signature lookupを含むlocal upper bound。
- overfitting risk: high。full arc-genとprivate-like holdout前のfamily分類である。

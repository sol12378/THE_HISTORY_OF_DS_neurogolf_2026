# exp018_neurogolf_dsl_core

## Purpose

7600を見据えて、全400タスクをprogram synthesisの対象として棚卸しする。各taskに対して、現cost、現source、route prediction、DSL family、形状/色/変更率/線構造特徴を保存する。

## Result

- base: `experiments/exp016_top100_rewrite_campaign`
- local estimate from base: `6479.383080`
- gap to 6500: `20.616920`
- gap to 7000: `520.616920`
- gap to 7600: `1120.616920`
- task registry: `experiments/exp018_neurogolf_dsl_core/task_registry.csv`
- primitive catalog: `experiments/exp018_neurogolf_dsl_core/primitive_catalog.md`

## Family Counts

- `signature_lookup_current`: 196
- `crop_resize`: 85
- `sparse_edit_or_object_completion`: 57
- `point_to_line_pattern`: 28
- `region_partition_fill`: 15
- `line_grid_fill`: 14
- `color_map`: 4
- `same_shape_global_transform`: 1

## Interpretation

7600への主戦場は、現在のsignature lookup依存を低コストでprivate耐性のある静的ONNX programへ置換すること。公開artifactの追加取得だけでは飽和している。

6500突破の短期優先は、`task187`, `task198`, `task203`, `task313` などの領域/線グリッド系と、`task398`, `task107` などのcrop/resize系。これらを数件だけ大きく圧縮できれば6500に届く。

## Next PDCA

1. `boundary_flood_fill`: 境界から到達可能な背景をunrolled Conv/Whereで求め、内外領域を塗り分ける。
2. `rectangular_room_fill`: 線グリッドの部屋構造を検出し、部屋単位で色を決める。
3. `point_to_line_pattern`: sparse seedから線、ray、box、反復motifを生成する。
4. `object_anchor_crop`: bbox/largest component/color maskからcrop/resizeを静的に表現する。
5. `static_lookup_compression`: signature lookupのfeature数/edits数を削減し、現lookupより低cost化する。

## Risk

- leakage risk: high。exp016 baseはlocal upper boundで、signature lookupを含む。
- overfitting risk: high。family分類は実装前の優先順位付けであり、full arc-genとprivate-like holdoutが必要。


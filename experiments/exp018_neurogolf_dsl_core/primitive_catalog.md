# exp018 Primitive Catalog

7600を見据え、タスクを小さな合成primitiveへ分解するための初期カタログ。
ここでは提出ONNXの実装ではなく、探索・分類・優先順位付けの共通語彙を固定する。

| Primitive | Family | ONNX lowering | Leakage risk | Private risk | Description |
|---|---|---|---|---|---|
| `identity` | `same_shape_global_transform` | Identity | low | low | Return the input unchanged; useful as a baseline and pipeline sentinel. |
| `color_map` | `same_shape_global_transform` | Equal + Where per mapped color | low | medium | Apply a constant per-color remapping learned from training pairs. |
| `fixed_crop` | `crop_resize` | Slice with constant starts/ends | low | medium | Extract a fixed rectangle shared by all training examples. |
| `signature_lookup` | `lookup` | MatMul/Gather/ScatterND style lookup | high | high | Memorize known input signatures and scatter a fixed output. |
| `boundary_flood_fill` | `region_partition_fill` | Conv + Where unrolled for a fixed grid radius | low | medium | Fill background components by boundary reachability using unrolled dilation. |
| `rectangular_room_fill` | `region_partition_fill` | Equal + Reduce/Conv + Where with static iterations | low | medium | Detect line-grid rooms and recolor enclosed or exterior cells. |
| `point_to_line_pattern` | `sparse_edit_or_object_completion` | Constant kernels + Conv/Gather/Where | low | medium | Expand sparse seed points into lines, rays, boxes, or repeated motifs. |

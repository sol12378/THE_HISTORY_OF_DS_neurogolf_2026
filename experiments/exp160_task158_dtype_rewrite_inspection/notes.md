# exp160_task158_dtype_rewrite_inspection

## 目的

exp159で候補になったtask158について、full-grid Cast-to-FLOATの消費先を確認し、dtype post-pass rewriteが安全そうか判断する。

## 結果

- cast_to_float_fullgrid_count: `18`
- rewrite_hint_counts: `{'numeric_consumers_need_float': 18}`
- decision: task158 casts feed numeric consumers; direct dtype removal is unlikely safe

## リスク

low: graph structure inspection only.
low-to-medium: rewrite hints may not be valid transformations.

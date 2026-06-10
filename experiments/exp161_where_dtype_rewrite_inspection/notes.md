# exp161_where_dtype_rewrite_inspection

## 目的

Whereが多いdtype候補taskについて、boolish-to-FLOAT CastがWhere条件としてだけ使われる安全rewrite候補かを監査する。

## 結果

- tasks: `[206, 338, 328, 366]`
- hint_counts_by_task: `{'206': {'where_cond=BOOL x=FLOAT16(1, 1, 30, 1) y=FLOAT16()': 2, 'where_cond=BOOL x=FLOAT16(1, 1, 1, 30) y=FLOAT16()': 2, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16()': 2, 'where_cond=BOOL x=FLOAT16() y=FLOAT16(1, 1, 30, 30)': 1, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16(1, 1, 30, 30)': 1, 'possible_keep_bool_as_where_condition': 1}, '338': {'where_cond=BOOL x=UINT8(1, 1, 30, 30) y=UINT8()': 1, 'where_cond=BOOL x=UINT8() y=UINT8(1, 1, 30, 30)': 13, 'numeric_consumers_need_float': 1}, '328': {'where_cond=BOOL x=FLOAT16(1, 1, 30, 1) y=FLOAT16(1, 1, 1, 30)': 3, 'where_cond=BOOL x=FLOAT16(1,) y=FLOAT16()': 4, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16()': 1, 'possible_keep_bool_as_where_condition': 1}, '366': {'where_cond=BOOL x=FLOAT16(1, 10, 30, 30) y=FLOAT16()': 2, 'where_cond=BOOL x=FLOAT16(1, 10, 30, 30) y=FLOAT16(1, 10, 30, 30)': 1, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16()': 36, 'boolish_to_float_mixed_consumers': 2, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16(1, 1, 30, 30)': 36, 'where_cond=BOOL x=FLOAT16() y=FLOAT16(1, 1, 30, 30)': 2, 'where_cond=BOOL x=FLOAT16(1, 1, 30, 30) y=FLOAT16(1, 1, 1, 1)': 5}}`
- promising_rows: `2`
- decision: inspect possible Where-condition cast rewrites for tasks 206 328

## リスク

low: graph structure inspection only.
low-to-medium: rewrite hints may not be valid transformations.

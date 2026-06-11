# exp330_task185_compact_dynamic_full_candidate

## 目的

exp329 の dynamic bg axis selector を full candidate に接続する。exp204 の巨大 `row_templates` / `col_templates` を避けるため、`best_row/best_col -> start/spacing -> Tile` で `GatherElements` index を動的生成する。

## 結果

- rows: `{'compact_dynamic_float': {'validation': '267_pass_0_fail', 'status': 'no_cost_gain', 'cost': 59784}, 'compact_dynamic_uint8': {'validation': '267_pass_0_fail', 'status': 'no_cost_gain', 'cost': 95784}}`
- best_row: `compact_dynamic_float`
- baseline_cost: `59584`
- best_cost: `59784`
- best_reason: `candidate cost is not lower than baseline`

## 判断

No submit. Use validation/cost failure to decide whether to debug task185 output core or pivot within GridSample queue.

## Submission

`no_submit`

## Risk

- leakage risk: low: input-only bg detector, window selector, and homogeneous local core; no output lookup or public feedback.
- overfitting risk: medium: task-specific geometry and dynamic index construction are validated locally but not yet public-calibrated.

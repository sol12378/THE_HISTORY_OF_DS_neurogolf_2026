# exp308_farm_next_low_cost_primitive_cost

## Plan

Phase C supplier の次施策選択を機械可読にするため、`next_low_cost_primitive` に加えて、その根拠となる `cost_proxy` を出す。見込みコスト削減は直接なし。

## Do

`farm_runner.py` の `run_smoke()` に `next_low_cost_primitive_cost` を追加。

## Check

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- low_cost_guardrail_min_cost_by_primitive:
  - `grid_sample`: `9`
  - `recolor_direct`: `45`
  - `recolor_cast`: `141`

## Act

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

# exp307_farm_next_low_cost_primitive

## Plan

Phase C supplier の次施策選択を機械化するため、farm smoke に `next_low_cost_primitive` を追加する。見込みコスト削減は直接なし。`low_cost_guardrail_min_cost_by_primitive` から最小 cost lane を選ぶ。

## Do

`farm_runner.py` の `run_smoke()` に `next_low_cost_primitive` を追加。

## Check

- low_cost_guardrail_min_cost_by_primitive:
  - `grid_sample`: `9`
  - `recolor_direct`: `45`
  - `recolor_cast`: `141`
- next_low_cost_primitive: `grid_sample`
- submitted_best_estimate: `6008.96`

## Act

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

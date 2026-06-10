# exp309_farm_next_low_cost_subjects

## Plan

Phase C supplier の次施策を具体化するため、`next_low_cost_primitive` に対応する subject を smoke summary に出す。見込みコスト削減は直接なし。

## Do

`farm_runner.py` の `run_smoke()` に `next_low_cost_subjects` を追加。

## Check

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- next_low_cost_subjects: `["smoke_gridsample"]`

## Act

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

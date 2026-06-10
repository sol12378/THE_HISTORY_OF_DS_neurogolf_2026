# exp307 farm next low cost primitive

## 目的

Phase C supplier の次施策選択を機械化するため、farm smoke に `next_low_cost_primitive` を追加する。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `low_cost_guardrail_min_cost_by_primitive` から最小 cost lane を選ぶ `next_low_cost_primitive` を追加。

## 結果

- `grid_sample`: `9`
- `recolor_direct`: `45`
- `recolor_cast`: `141`
- next_low_cost_primitive: `grid_sample`

## 判断

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

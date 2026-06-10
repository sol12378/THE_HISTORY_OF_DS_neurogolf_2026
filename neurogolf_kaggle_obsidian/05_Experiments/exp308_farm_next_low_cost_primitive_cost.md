# exp308 farm next low cost primitive cost

## 目的

Phase C supplier の次施策選択を機械可読にするため、`next_low_cost_primitive` に加えて根拠 cost を出す。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `run_smoke()` に `next_low_cost_primitive_cost` を追加。

## 結果

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- `recolor_direct`: `45`
- `recolor_cast`: `141`

## 判断

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

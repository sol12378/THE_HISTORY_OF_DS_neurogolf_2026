# exp309 farm next low cost subjects

## 目的

Phase C supplier の次施策を具体化するため、`next_low_cost_primitive` に対応する subject を smoke summary に出す。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `run_smoke()` に `next_low_cost_subjects` を追加。

## 結果

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- next_low_cost_subjects: `["smoke_gridsample"]`

## 判断

smoke planner metadata のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke planner metadata の追加のみ。

# exp313 farm notes next low cost

## 目的

Phase C の次手を notes から追えるように、farm notes に `next_low_cost_primitive` と根拠 cost を出す。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `write_notes()` に `next low-cost primitive`, `next low-cost primitive cost`, `next low-cost subjects` を追加。

## 結果

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- next_low_cost_subjects: `["smoke_gridsample"]`
- notes_has_grid_sample: `true`
- notes_has_cost: `true`

## 判断

notes visibility のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `grid_sample` supplier を実task候補へ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。notes visibility の追加のみ。

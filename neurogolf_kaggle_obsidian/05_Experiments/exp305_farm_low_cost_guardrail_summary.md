# exp305 farm low cost guardrail summary

## 目的

Phase C supplier の lane review を速くするため、farm smoke の結果に low-cost subject summary を追加する。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `run_smoke()` に `low_cost_guardrail_subjects` を追加。

## 結果

- low_cost_guardrail_subjects: `["smoke_recolor_direct", "smoke_recolor_cast", "smoke_gridsample"]`
- grid_sample_visible: `true`
- submitted_best_estimate: `6008.96`

## 判断

smoke summary のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke summary の追加のみ。

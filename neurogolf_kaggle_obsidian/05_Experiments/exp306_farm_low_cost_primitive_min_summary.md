# exp306 farm low cost primitive min summary

## 目的

Phase C supplier の優先順位付けに向け、farm smoke に primitive kind ごとの最小 `cost_proxy` を出す。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `run_smoke()` に `low_cost_guardrail_min_cost_by_primitive` を追加。

## 結果

- `grid_sample`: `9`
- `recolor_direct`: `45`
- `recolor_cast`: `141`
- submitted_best_estimate: `6008.96`

## 判断

smoke summary のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke summary の追加のみ。

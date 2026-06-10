# exp306_farm_low_cost_primitive_min_summary

## Plan

Phase C supplier の優先順位付けに向け、farm smoke に primitive kind ごとの最小 `cost_proxy` を出す。見込みコスト削減は直接なし。`grid_sample` と `recolor_direct` のような lane を即比較するための可視化。

## Do

`farm_runner.py` の `run_smoke()` に `low_cost_guardrail_min_cost_by_primitive` を追加。

## Check

- low_cost_guardrail_min_cost_by_primitive:
  - `grid_sample`: `9`
  - `recolor_direct`: `45`
  - `recolor_cast`: `141`
- submitted_best_estimate: `6008.96`

## Act

smoke summary のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke summary の追加のみ。

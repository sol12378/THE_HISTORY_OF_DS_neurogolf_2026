# exp305_farm_low_cost_guardrail_summary

## Plan

Phase C supplier の lane review を速くするため、farm smoke の結果に `low_cost_guardrail_subjects` を追加する。見込みコスト削減は直接なし。実task emission 前の採用可能 lane を素早く確認するための可視化。

## Do

`farm_runner.py` の `run_smoke()` に `low_cost_guardrail_subjects` を追加。

## Check

- low_cost_guardrail_subjects: `["smoke_recolor_direct", "smoke_recolor_cast", "smoke_gridsample"]`
- grid_sample_visible: `true`
- submitted_best_estimate: `6008.96`

## Act

smoke summary のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke summary の追加のみ。

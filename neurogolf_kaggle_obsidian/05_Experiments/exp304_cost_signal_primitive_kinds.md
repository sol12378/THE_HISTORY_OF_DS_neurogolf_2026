# exp304 CostSignal primitive kinds

## 目的

Phase C supplier の lane 別分析に向け、farm guardrail row に `primitive_kinds` を出す。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py`
- `CostSignal` に `primitive_kinds` を追加。
- `assess_ir()` で `program.primitive_kinds` を設定。

## 結果

- primitive_kinds: `["grid_sample"]`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`

## 判断

tooling metadata の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。signal metadata の追加のみ。

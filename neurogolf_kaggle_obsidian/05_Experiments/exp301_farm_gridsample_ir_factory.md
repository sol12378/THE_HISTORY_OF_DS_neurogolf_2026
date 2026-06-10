# exp301 farm GridSample IR factory

## 目的

Phase C の `one_node_gridsample` を実 supplier へ接続するため、farm IR に標準 factory を追加する。

## 変更

- `experiments/neurogolf_farm/ir.py`
- `gridsample_program()` を追加。
- default は `output_dtype="uint8"`, `param_count=8`。

## 結果

factory 経由の synthetic IR:

- subject: `smoke_gridsample_factory`
- op_type: `GridSample`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`

## 判断

tooling factory の追加のみで実候補bundleのlocal estimate改善ではないため提出なし。次は `farm_runner.py` smoke visibility か実task supplier接続へ進む。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。IR factory の追加のみ。

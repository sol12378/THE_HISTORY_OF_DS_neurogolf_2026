# exp303 farm GridSample primitive kind

## 目的

Phase C の `one_node_gridsample` lane を他の one-node data movement と切り分けるため、専用 primitive kind を追加する。

## 変更

- `experiments/neurogolf_farm/ir.py`
- `PrimitiveKind.GRID_SAMPLE = "grid_sample"` を追加。
- `gridsample_program()` の node kind を `GRID_SAMPLE` に変更。

## 結果

- primitive_kinds: `("grid_sample",)`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`

## 判断

tooling label の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。IR label の追加のみ。

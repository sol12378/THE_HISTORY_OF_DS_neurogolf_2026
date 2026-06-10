# exp300 farm GridSample low cost gate

## 目的

roadmap Phase C の `one_node_gridsample` 逆設計へ戻るため、farm cost gate が `GridSample` を fused low-cost data movement として扱えるか確認する。

## 変更

- `experiments/neurogolf_farm/cost_extractor.py`
- `LOW_COST_OPS` に `GridSample` を追加。

## 結果

synthetic IR:

- op_type: `GridSample`
- param_count: `8`
- output: `uint8`, shape `(1, 1, 1, 1)`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`

## 判断

tooling gate の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は GridSample supplier/逆設計候補を実taskへ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。cost gate の表現追加のみ。

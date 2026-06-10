# exp302 farm GridSample smoke visibility

## 目的

Phase C の `one_node_gridsample` supplier 接続に向け、farm smoke で GridSample factory の cost を常時見えるようにする。

## 変更

- `experiments/neurogolf_farm/farm_runner.py`
- `_smoke_programs()` に `gridsample_program("smoke_gridsample", output_shape=(1, 1, 1, 1))` を追加。

## 結果

- subject: `smoke_gridsample`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`
- submitted_best_estimate: `6008.96`

## 判断

smoke visibility のみで実候補bundleのlocal estimate改善ではないため提出なし。次は GridSample supplier/逆設計候補を実taskへ接続する。

## リスク

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke visibility の追加のみ。

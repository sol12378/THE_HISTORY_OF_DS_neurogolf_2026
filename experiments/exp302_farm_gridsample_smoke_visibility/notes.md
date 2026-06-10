# exp302_farm_gridsample_smoke_visibility

## Plan

Phase C の `one_node_gridsample` supplier 接続に向け、farm smoke で GridSample factory の cost を常時見えるようにする。見込みは `param_count=8` + `uint8` output 1 byte = `cost_proxy=9`。

## Do

`farm_runner.py` の `_smoke_programs()` に `gridsample_program("smoke_gridsample", output_shape=(1, 1, 1, 1))` を追加。

## Check

- subject: `smoke_gridsample`
- cost_proxy: `9`
- predicted_cost_band: `250-600_plausible`
- hard_reject: `false`
- submitted_best_estimate: `6008.96`

## Act

smoke visibility のみで実候補bundleのlocal estimate改善ではないため提出なし。次は GridSample supplier/逆設計候補を実taskへ接続する。

## Risk

- leakage risk: low。提出候補なし。
- overfitting risk: low。farm smoke visibility の追加のみ。

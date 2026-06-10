# exp301_farm_gridsample_ir_factory

## Plan

Phase C の one_node_gridsample を実supplierへ接続する前に、標準IR factoryを追加する。見込みは param_count 8 + uint8 output bytes。

## Do

ir.py に gridsample_program() を追加。

## Check

- subject: smoke_gridsample_factory
- cost_proxy: 9
- predicted_cost_band: 250-600_plausible
- hard_reject: False

## Act

tooling factory の追加のみで実候補bundleのlocal estimate改善ではないため提出なし。次は arm_runner.py smoke visibility か実task supplier接続へ進む。

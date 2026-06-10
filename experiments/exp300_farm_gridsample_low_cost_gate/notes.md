# exp300_farm_gridsample_low_cost_gate

## Plan

Phase C の低cost fused loweringへ戻る。one-node GridSample archetype を farm cost gate で低cost候補として扱えるようにする。見込みは param_count 8 + uint8 output 1 = cost_proxy 9。

## Do

cost_extractor.py の LOW_COST_OPS に GridSample を追加。

## Check

- subject: smoke_one_node_gridsample
- cost_proxy: 9
- predicted_cost_band: 250-600_plausible
- hard_reject: False

## Act

tooling gate の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は GridSample supplier/逆設計候補を実taskへ接続する。

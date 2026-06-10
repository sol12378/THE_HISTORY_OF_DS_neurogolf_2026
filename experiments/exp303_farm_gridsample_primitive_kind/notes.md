# exp303_farm_gridsample_primitive_kind

## Plan

Phase C の one_node_gridsample lane を他の one-node data movement と切り分けるため、専用 primitive kind を追加する。見込みコスト削減は直接なし。実task supplier接続時の分析/採否を明確化する。

## Do

ir.py に PrimitiveKind.GRID_SAMPLE を追加し、gridsample_program() の node kind を差し替え。

## Check

- primitive_kinds: ('grid_sample',)
- cost_proxy: 9
- predicted_cost_band: 250-600_plausible
- hard_reject: False

## Act

tooling label の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

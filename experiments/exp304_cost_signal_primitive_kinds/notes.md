# exp304_cost_signal_primitive_kinds

## Plan

Phase C supplier の lane 別分析に向け、CostSignal.as_dict() に primitive_kinds を出す。見込みコスト削減は直接なし。

## Do

cost_extractor.py の CostSignal に primitive_kinds を追加し、ssess_ir() で program.primitive_kinds を設定。

## Check

- primitive_kinds: ('grid_sample',)
- cost_proxy: 9
- predicted_cost_band: 250-600_plausible
- hard_reject: False

## Act

tooling metadata の更新のみで実候補bundleのlocal estimate改善ではないため提出なし。次は実task supplier接続へ進む。

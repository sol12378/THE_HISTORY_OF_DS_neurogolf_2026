# exp314_grid_sample_priority_targets

## Plan

Phase C の `grid_sample` lane を実 task へ接続する前段として、farm result に優先 target queue と cost 600 化した場合の期待 delta を出す。直接の cost 削減はまだないが、次の supplier 実装を高期待値 task へ固定する。

## Do

`farm_runner.py` の `run_smoke()` に `grid_sample_priority_targets` を追加し、`write_notes()` にも表示した。

## Check

- next_low_cost_primitive: `grid_sample`
- next_low_cost_primitive_cost: `9`
- target_count: `4`
- target 251: current_cost `100580`, floor_cost `600`, expected_delta `5.121876`
- target 037: current_cost `63726`, floor_cost `600`, expected_delta `4.665371`
- target 185: current_cost `59584`, floor_cost `600`, expected_delta `4.598086`
- target 048: current_cost `5481`, floor_cost `600`, expected_delta `2.212843`

## Act

smoke/queue visibility のみで実候補 bundle の local estimate 改善ではないため提出なし。次は task251 または task037 に対して `grid_sample` supplier の最小 real-candidate 生成を試す。

## Risk

- leakage risk: low。roadmap と既知 cost からの target queue のみ。
- overfitting risk: low。Kaggle candidate は生成していない。

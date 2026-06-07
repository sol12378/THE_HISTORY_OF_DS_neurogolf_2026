# exp019_boundary_flood_fill_campaign

## Purpose

`task187` などの背景0を壁で分割するtaskに対して、境界から到達可能な外側背景と閉領域背景を静的ONNXで塗り分ける `boundary_flood_fill` を試す。

## Result

- base: `experiments/exp016_top100_rewrite_campaign`
- top_k: `80`
- arc_gen_sample: `20`
- baseline local estimate: `6479.383086`
- new local estimate: `6479.383086`
- delta: `0.000000`
- improved tasks: `0`

## Finding

`task187` はルールとしては `boundary_flood_fill` で正答できた。18 step以上では `24_pass_0_fail` になったが、costが `237631` 以上になり、baseline cost `105313` より悪化した。

つまり問題は推論規則ではなくONNX loweringのmemory cost。unrolled flood fillは正しいが、このままでは6500突破の採用候補にならない。

## Next

優先度を `diagonal_shift_tile` (`task398`), `periodic_shift_fill` (`task313`, `task221`), `ring_depth_remap` (`task203`) に移す。これらは反復floodより少ない演算で表せる可能性が高い。

## Risk

- leakage risk: high。baseはexp016 local upper bound。
- overfitting risk: high。sample-local評価でありfull arc-gen前。


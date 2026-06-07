# exp020_diagonal_periodic_lowmem

## Purpose

`task313` の周期パターン補完を、static ONNX programとして実装し、exp016の高cost baselineより低costにできるか確認する。

## Result

- base: `experiments/exp016_top100_rewrite_campaign`
- task: `313`
- arc_gen_sample: `20`
- baseline local estimate: `6479.383086`
- new local estimate: `6479.383086`
- delta: `0.000000`
- improved tasks: `0`

## Candidate Detail

- candidate: `periodic_shift_fill_p2_p3`
- validation: `24_pass_0_fail`
- candidate cost: `215479`
- baseline cost: `83572`
- decision: `no_cost_gain`

## Interpretation

規則自体は正しい。`task313` は「左上の2行×period列を、列方向に1つずらして全体へ敷く」周期補完だった。

ただし、`Where` を複数回重ねるONNX loweringは中間tensor memoryが大きく、baselineより高costになった。次はfull-grid `Where` ではなく、更新点だけを持つ `ScatterND`、または既存artifactのgraph surgeryでcostを削る。

## Next

- `task398`: `diagonal_shift_tile` を sparse ScatterND で試す。
- `task203`: `ring_depth_remap` をdepth maskではなく、既存artifact pruning/surgeryで圧縮する。
- signature lookup系: feature数とupdate数を減らすcompressionを優先する。

## Risk

- leakage risk: high。baseはexp016 local upper bound。
- overfitting risk: high。sample-local評価でありfull arc-gen前。


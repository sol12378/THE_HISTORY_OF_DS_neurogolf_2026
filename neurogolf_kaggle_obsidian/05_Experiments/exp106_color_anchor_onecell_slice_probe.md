# exp106_color_anchor_onecell_slice_probe

## Hypothesis

exp105で全nonzero bbox-min anchorが否定されたため、1x1 P0 output taskでは色別bbox-min anchorを使えば、marker/object起点の1セル抽出として説明できる可能性がある。

ONNX候補は `Slice(channel c) -> ReduceSum -> ArgMax -> Slice(one cell) -> Pad`。

## Result

- campaign #: `8`
- base local estimate: `6282.812218`
- targets: `4`
- color-anchor fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`
- submission decision: `no_submit`

## Interpretation

task355/346/48/291は、単純な色別marker近傍の1セル抽出では説明できなかった。1x1 laneはまだ出力floorが軽いが、一般anchor探索を広げるより、task別にbridge/connectivity/object relationなどのrule診断へ戻すべき。

## Risk

- leakage risk: medium。
- overfitting risk: medium。今回candidateは出ていない。

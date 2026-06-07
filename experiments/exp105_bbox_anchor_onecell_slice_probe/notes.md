# exp105_bbox_anchor_onecell_slice_probe

## Hypothesis

After exp104 rejected absolute coordinates, a small dynamic anchor may be enough for 1x1 P0 outputs. Candidate lowering computes the nonzero bbox top-left with `ReduceSum + ArgMax`, slices one cell at a learned offset, and pads it back to official output shape.

## Result

- targets: `4`
- bbox-min offset fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Interpretation

This is a real `tiny_dynamic_shape_index` candidate-emission probe. If the candidates validate but lose on cost, the next step is to mine the exact low-cost artifact structure rather than adding more generic ArgMax/Slice blocks.

## Risk

- leakage risk: medium。
- overfitting risk: medium。offset ruleの視覚的妥当性を確認してからsubmit判断する。

# exp106_color_anchor_onecell_slice_probe

## Hypothesis

exp105 showed that all-nonzero bbox-min is too coarse for 1x1 P0 outputs. A color-specific bbox-min anchor may isolate the relevant marker/object and allow a small `Slice(channel c) -> ArgMax -> Slice(one cell) -> Pad` lowering.

## Result

- targets: `4`
- color-anchor fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Interpretation

This probes a concrete `tiny_dynamic_shape_index` variant. If fit remains 0, 1x1 P0 outputs are not simple marker-neighbor extraction and should move to rule-specific diagnostics.

## Risk

- leakage risk: medium。
- overfitting risk: medium。色+offsetの小ruleでも、視覚的説明がない場合はsubmitしない。

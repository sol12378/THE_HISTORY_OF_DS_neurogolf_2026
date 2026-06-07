# exp104_p0_fixed_3x3_static_index_probe

## Hypothesis

P0 small-output tasks may hide a very cheap `one_node_gather_index_map` / `computed_slice_pad` style lowering: if every output cell is copied from a fixed input coordinate across all examples, `Reshape -> Gather(axis=2) -> Reshape -> Pad` can replace expensive crop/shape artifacts.

## Result

- targets: `37`
- fixed-position fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Interpretation

This is an actual candidate-emission experiment, not a catalog. If accepted is empty, fixed absolute index maps are too brittle or already beaten by current artifacts; the next compiler lane should make the index dynamic from shape/object anchors rather than hard-code coordinates.

## Risk

- leakage risk: medium。
- overfitting risk: medium-to-high。acceptedが出ても、座標が視覚的ruleで説明できるか確認してからsubmit判断する。

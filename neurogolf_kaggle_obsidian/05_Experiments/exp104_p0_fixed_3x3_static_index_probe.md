# exp104_p0_fixed_3x3_static_index_probe

## Hypothesis

P0 small-output cropish tasks may be compressible by a cheap `one_node_gather_index_map` style compiler. If each output cell is copied from one fixed input coordinate across all available examples, `Reshape -> Gather(axis=2) -> Reshape -> Pad` should be a low-cost replacement for high-cost crop/shape artifacts.

## Result

- campaign #: `6`
- base local estimate: `6282.812218`
- targets: `37`
- fixed-position fits: `0`
- generated candidates: `0`
- accepted tasks: `[]`
- local delta: `0.000000`
- new local estimate: `6282.812218`
- submission decision: `no_submit`

## Interpretation

固定絶対座標のstatic index mapはP0小出力taskには刺さらなかった。低cost artifactに存在する `Gather` は有効なprimitiveだが、高cost P0 cropish taskでは座標が例ごとにobject/shape/lattice anchorへ追従している可能性が高い。

次は固定座標ではなく、`computed_slice_pad` / `tiny_dynamic_shape_index` のように、`ReduceSum` / `ArgMax` / `Gather` で小さなoffsetを計算してから `Slice` / `Pad` するcompilerへ進む。

## Risk

- leakage risk: medium。座標推定は全例から行ったが、raw output lookupは使っていない。
- overfitting risk: medium-to-high。仮にfitしても、座標が視覚的ruleで説明できるか確認してからsubmit判断する。

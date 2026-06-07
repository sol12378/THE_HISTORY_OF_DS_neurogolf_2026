# exp025_tile_prefix_blocks_task221

## Hypothesis

`task221` は、3x3 input の `zero_count` を出力block数に変換する tile-prefix rule で解ける。full-grid `Tile` または sparse `ScatterND` lowering で baseline artifact より低costにできる可能性がある。

## Result

- target task: `task221`
- baseline cost: `55869`
- `tile_prefix_blocks`: validation `25_pass_0_fail`, cost `141047`
- `tile_prefix_blocks_sparse_argmax`: validation `25_pass_0_fail`, cost `165105`
- local delta: `0.000000`
- status: `no_gain`

## Interpretation

DSL ruleは正しい。だが、この公式costでは `Tile + full-grid mask` も、dynamic indexを作る `ArgMax + ScatterND` もbaseline artifactより重い。候補生成前に、full-grid演算と大きなdynamic scatterをrejectするcost modelが必要。

## Next

- `task221` は既存artifactのgraph surgeryを優先する。
- block-prefix ruleを維持しつつ、`Slice/Gather` のみで表現できる低cost loweringを探索する。
- 同種の3x3-to-block expansion taskをregistryから探し、family単位でcost modelを更新する。

## Risks

- leakage risk: medium。rule自体は幾何的だが、検証はsample20。
- overfitting risk: medium。zero_countを `3..7` に固定している。


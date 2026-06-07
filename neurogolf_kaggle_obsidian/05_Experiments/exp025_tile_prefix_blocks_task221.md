# exp025_tile_prefix_blocks_task221

## Hypothesis

task221 は、3x3 input の `zero_count` から出力サイズとprefix block数を決める rule で解ける。低cost loweringにできれば、baseline cost `55869` を下げられる。

## Result

- target task: `task221`
- baseline cost: `55869`
- `tile_prefix_blocks`: validation `25_pass_0_fail`, cost `141047`
- `tile_prefix_blocks_sparse_argmax`: validation `25_pass_0_fail`, cost `165105`
- status: `no_gain`

## Interpretation

ruleは正しいが、loweringが重い。full-grid `Tile + mask` はmemory costが高く、sparse版も `ArgMax + ScatterND` のdynamic index構築が重くなる。

## Decision

task221は次に、既存artifactのgraph surgery、または `Slice/Gather` 中心のloweringを検討する。`Tile` と大きな `ScatterND` はcost modelで事前に警戒する。

## Risks

- leakage risk: medium
- overfitting risk: medium


# exp_b018_high_precision_island_or_tree_miner

## 目的

exp_b017で見つかったhigh precision feature islandをOR treeとして足し、train full fitする説明可能branch ruleを探索する。

## 結果

- target tasks: `5`
- evaluated candidates: `4`
- full pass: `0`
- train/test pass: `0`
- train-fit: `0`
- best coverage:
  - task286: 10 branches, train positive coverage `10/92`, false positive `0`
  - task133: 3 branches, coverage `6/89`, false positive `0`
  - task285: 6 branches, coverage `6/81`, false positive `0`
  - task173: 1 branch, coverage `1/23`, false positive `0`
- submission: `no_submit`

## 解釈

precision 1.0 islandは存在するが、単独feature値のORではcoverageが低すぎる。L2 sparse fillは、単独feature値ではなく、pairwise conjunctionやcolor-role/component-roleを含むbranchが必要。

## Decision

L2の次手はpairwise feature conjunction tree。ただしb014-b018はscore deltaが出ていないため、次PDCAは一度、提出候補を作りやすいlaneへ寄せる。

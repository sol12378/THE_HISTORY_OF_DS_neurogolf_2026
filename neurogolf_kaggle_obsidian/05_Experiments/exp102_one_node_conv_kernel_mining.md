# exp102_one_node_conv_kernel_mining

## 目的

exp100で最大archetypeだった `one_node_conv_kernel` を逆解析し、Convをcompiler primitiveとして使う条件とguardrailを作る。

## 結果

- conv artifacts: `23`
- conv class counts:
  - `sparse_local_pattern_detector`: 16
  - `general_conv_kernel`: 6
  - `depthwise_local_filter`: 1
- local delta: `0.000000`
- new local estimate: `6282.812218`

## Decision

次のscore-producing候補は、connectivity/rayではなく、task185型の小lattice/local-maskに対して one Conv detector を使う方向にする。巨大visibility Convはguardrailで弾く。

## Risk

- leakage risk: low。
- overfitting risk: catalog段階ではlow。

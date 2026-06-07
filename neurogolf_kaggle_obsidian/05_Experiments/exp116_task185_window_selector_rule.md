# exp116_task185_window_selector_rule

## Hypothesis

task185のlattice位置はraw coordinate tableではなく、grid line上の4連続windowをscoreして選べる。

## Result

- campaign index: `18`
- pass: `267/267`
- fail: `0`
- local delta: `0.000000`

## Selector

1. background colorを最頻非zero色として推定する。
2. background比率が高い行/列をgrid line候補にする。
3. spacing 3/4/5 の4連続row window / col windowを列挙する。
4. 各window pairについて、交点16個のうち非zeroかつ非backgroundの数をscoreにする。
5. 最高scoreの4x4 matrixを使い、homogeneous nonzero 2x2 blockを3x3へ圧縮する。

## Interpretation

task185はraw position tableなしで解ける。exp115の46 raw position patternは、window scoringで吸収できる。これはLB上も重要で、hidden汎化しやすいinput-only geometry compilerとして扱える。

次の課題はONNX lowering。既にexp114でcoreはcost `1516` まで測れているため、window scoring/selectionをどれだけ低costにできるかが勝負。

## Risk

- leakage risk: low。input-only geometry selector。
- overfitting risk: medium-low。raw table化せずwindow scoringとして実装する限り許容。

# exp055_l1_sparse_fill_rule_miner

## Hypothesis

L1 static sparse background fill 8 taskは、行列交点、近傍数、対角線、bbox、D4 orbitなどの低cost ruleで説明できる。

## Result

- scanned tasks: `8`
- candidate rules: `30`
- evaluated candidates: `240`
- full pass hits: `0`
- train/test pass hits: `0`
- best partial: task020 `d4_orbit_complete`, `132/266`

## Interpretation

単純な幾何/local ruleだけではL1でも不足。task020のD4 orbitは有効な部分構造だが、missingが残る。

次はobject-role predicate、local frame、色roleの条件を導入する。full pass hitが出るまではONNX loweringしない。

## Risk

- leakage risk: low。汎用ruleのみ。
- overfitting risk: medium。all arc-genを診断に使っているため、採用候補にはholdoutと小さいKaggle delta較正が必要。

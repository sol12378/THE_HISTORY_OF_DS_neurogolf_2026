# exp_b002_p0_explainable_rule_sweep

## 目的

queue上位P0 signature taskに対して、D4 orbit、rectangle closure、row/column segment fillのような説明可能ruleが効くか確認する。

## 結果

- scanned tasks: `40`
- evaluated rules: `120`
- full pass hits: `0`
- train/test pass hits: `0`

## 判断

単純幾何ruleはqueue上位signature taskには弱い。また、このP0定義はteacher-gain P0とずれていたため、次はexp048のteacher-gain P0を対象にする。

## Risk

- leakage risk: 低。teacher label tableは使っていない。
- overfitting risk: 中。all arc-gen診断だが採用候補なし。

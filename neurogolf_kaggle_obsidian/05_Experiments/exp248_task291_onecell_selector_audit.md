# exp248_task291_onecell_selector_audit

## 目的

task291は1x1固定・binary signature 1種類なので、出力色がcount/bbox/componentなどの単純selectorで決まるか監査する。

## 結果

- baseline_cost: `4327`
- example_count: `265`
- target_count_rank_hist: `{0: 37, 1: 61, 2: 76, 3: 48, 4: 27, 5: 14, 6: 2}`
- best rule: `count_rank2 76/265`

## 判断

単純selectorは弱い。task291は保留。

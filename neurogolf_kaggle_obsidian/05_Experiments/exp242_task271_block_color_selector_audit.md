# exp242_task271_block_color_selector_audit

## 目的

task271の4つのfull-nonzero 3x3 blockから、正解blockを色構成rankで選べるか確認する。

## 結果

- `color8_count_min`: `267/267`
- `sum_colors_min`: `267/267`
- 次点 `edge_sum_min`: `204/267`
- target blockは常に2色で、color8 countが4候補中最小。

## 判断

task271 ruleを発見。入力内の3x3 full-nonzero blockのうち、color8_count最小のblockを出力する。次はONNX cost probe。

## リスク

tie-break edgeはhiddenで確認が必要。ただし単純な入力構造ruleであり、lookupではない。

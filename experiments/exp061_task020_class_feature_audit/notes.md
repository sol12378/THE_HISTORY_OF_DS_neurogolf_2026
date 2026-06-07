# exp061_task020_class_feature_audit

## 目的

task020の3 template classを入力特徴から選ぶため、feature keyごとの純度を調べる。

## 結果

- examples: 266
- template counts: {'corners': 87, 'inner_diag': 87, 'edge_mid': 92}
- best key: {'feature_key': 'pos', 'unique_key_count': 71, 'pure_key_count': 71, 'covered_examples': 266, 'majority_correct': 266, 'accuracy': 1.0, 'top_conflicts': '[]'}

## Key Summary

| key | accuracy | unique keys | pure keys |
|---|---:|---:|---:|
| pos | 1.000 | 71 | 71 |
| row_col_counts | 1.000 | 71 | 71 |
| canon_pos | 1.000 | 24 | 24 |
| canon_and_count | 1.000 | 24 | 24 |
| col_counts | 0.876 | 40 | 34 |
| row_counts | 0.872 | 41 | 35 |
| count_and_touch | 0.820 | 14 | 6 |
| touch_signature | 0.797 | 8 | 2 |
| touches_edge_mid | 0.617 | 2 | 0 |
| touches_corner | 0.590 | 2 | 0 |
| sorted_col_counts | 0.489 | 16 | 10 |
| sorted_row_counts | 0.466 | 15 | 8 |

## Decision

高cardinality lookupは使わない。低cardinalityで高精度な特徴があれば、rule selectorへ落とす。

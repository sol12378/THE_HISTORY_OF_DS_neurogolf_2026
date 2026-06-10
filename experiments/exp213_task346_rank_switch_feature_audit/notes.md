# exp213_task346_rank_switch_feature_audit

## 目的

task346で`least_nz`が失敗する4例をrank0/rank1切替として説明できる簡単な特徴を探す。

## 結果

- switch_count: `4`
- best_predicates: `[{'key': 'd_largest_component', 'op': '<=', 'thr': -5, 'tp': 4, 'fp': 0, 'fn': 0}, {'key': 'r0_largest_component', 'op': '>=', 'thr': 8, 'tp': 4, 'fp': 0, 'fn': 0}, {'key': 'r1_largest_component', 'op': '<=', 'thr': 3, 'tp': 4, 'fp': 0, 'fn': 0}, {'key': 'd_largest_component', 'op': '<=', 'thr': 4, 'tp': 4, 'fp': 2, 'fn': 0}, {'key': 'd_component_count', 'op': '>=', 'thr': 5, 'tp': 4, 'fp': 5, 'fn': 0}]`

## 判断

低FPの単純predicateがあればrank-switch ruleを検証する。なければpivot。

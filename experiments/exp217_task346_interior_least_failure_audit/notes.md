# exp217_task346_interior_least_failure_audit

## 目的

task346の `interior_least_nz` が失敗する2例を監査し、component growthなしの安い補正で267/267へ到達できるかを判断する。

## 結果

- pass/fail: `265/2`
- failures: `[(89, 6, 1, [(1, 14), (6, 14)], [(1, 3), (6, 3)], [0, 0, 1, 0]), (95, 8, 4, [(4, 10), (8, 11)], [(8, 2), (4, 9)], [0, 0, 0, 0])]`
- best_simple_branch_tests: `[{'feature': 'all_count', 'op': '>=', 'threshold': 17, 'tp': 2, 'fp': 4, 'fn': 0}, {'feature': 'all_count', 'op': '>=', 'threshold': 16, 'tp': 2, 'fp': 6, 'fn': 0}, {'feature': 'all_count', 'op': '>=', 'threshold': 15, 'tp': 2, 'fp': 9, 'fn': 0}, {'feature': 'all_count', 'op': '>=', 'threshold': 14, 'tp': 2, 'fp': 13, 'fn': 0}, {'feature': 'interior_count', 'op': '>=', 'threshold': 10, 'tp': 2, 'fp': 13, 'fn': 0}]`

## 判断

TP2 FP0の単純count系分岐があれば次はONNX cost probe。なければexp214同様component条件が必要で、exp215のcost wallから一旦保留。

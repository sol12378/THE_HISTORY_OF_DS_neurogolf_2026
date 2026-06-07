# exp055_l1_sparse_fill_rule_miner

## 目的

L1 static sparse background fill 8 taskに対し、低cost lowering可能な明示ruleを探索する。

## 結果

- scanned tasks: 8
- evaluated rules: 240
- full pass hits: 0
- train/test pass hits: 0

## Best Partial

- {'task_id': 20, 'rule_name': 'd4_orbit_complete', 'status': 'partial', 'total_pass': 132, 'total_examples': 266, 'train_pass': 2, 'train_examples': 3, 'test_pass': 1, 'test_examples': 1, 'arc_pass': 129, 'arc_examples': 262, 'fail_reasons': '{"missing=3;extra=0;wrong=0": 134}', 'estimated_lowering': 'Equal color masks + small Reduce/Conv/constant masks + one Where', 'target_cost_band': '250-600 if rule is full-pass and emitted without full-grid tables'}

## 解釈

full pass hitがあれば、次はそのruleだけを小さいONNXへloweringする。hitがなければ、L1でも単純な行列/近傍/対称性だけでは不足なので、object-role predicateを追加する。

# exp057_l1_object_role_feature_miner

## 目的

L1 sparse background fill taskに対して、object-role / bbox-local boolean featureのAND ruleを探索する。

## 結果

- scanned tasks: 8
- evaluated candidates: 220
- full pass hits: 0
- train/test pass hits: 0
- best overall: {'task_id': 50, 'rule_name': 'and_features', 'status': 'partial', 'total_pass': 6, 'total_examples': 271, 'train_pass': 1, 'train_examples': 8, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 5, 'arc_examples': 262, 'predicted_change_median': 27.0, 'fail_reasons': '{"missing=0;extra=0;wrong=1": 10, "missing=0;extra=0;wrong=2": 8, "missing=0;extra=4;wrong=7": 6, "missing=0;extra=0;wrong=3": 4, "missing=0;extra=5;wrong=6": 4}', 'rule_spec': '{"all_features": ["allbbox_inside"]}'}

## 解釈

full-pass single-task ruleが出た場合のみ、低cost ONNXへloweringしてKaggle single-task delta提出へ進む。

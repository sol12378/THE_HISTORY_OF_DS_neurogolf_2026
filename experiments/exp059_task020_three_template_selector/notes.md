# exp059_task020_three_template_selector

## 目的

task020を3-template selector問題として解く。

## 結果

- selector count: 5
- full pass hits: 0
- best: {'selector_name': 'touch_rule', 'status': 'partial', 'total_pass': 142, 'total_examples': 266, 'train_pass': 2, 'train_examples': 3, 'test_pass': 1, 'test_examples': 1, 'arc_pass': 139, 'arc_examples': 262, 'template_accuracy': 142, 'template_total': 266, 'fail_reasons': '{"no_template": 86, "template_ok_but_apply_fail": 86, "wrong_template:corners->inner_diag": 17, "wrong_template:edge_mid->inner_diag": 13, "wrong_template:inner_diag->edge_mid": 4, "wrong_template:inner_diag->corners": 4}', 'selector_spec': '{"kind": "touch_rule"}'}

## Decision

full passが出るまではONNX loweringしない。train lookupでよい結果が出ても、そのままtemplate tableとして提出しない。

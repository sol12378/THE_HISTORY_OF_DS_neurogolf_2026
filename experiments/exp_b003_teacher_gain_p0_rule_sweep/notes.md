# exp_b003_teacher_gain_p0_rule_sweep

## 目的

exp_b002ではqueue上位P0を対象にしてしまい、teacher-gain P0とずれていた。今回はexp048のteacher-gain P0 18 taskを対象に、同じ説明可能rule familyを評価する。

## 仮説

teacherが大きく改善したP0 taskは、queue上位signature lookupよりも説明可能rule hitが出やすい。

## 結果

- scanned tasks: 18
- evaluated rows: 54
- full pass hits: 0
- train/test pass hits: 0
- best partial: {'task_id': 20, 'rank': 1, 'family': 'sparse_edit_or_object_completion', 'rule_name': 'square_d4_orbit_completion', 'status': 'partial', 'total_pass': 132, 'total_examples': 266, 'train_pass': 2, 'train_examples': 3, 'test_pass': 1, 'test_examples': 1, 'arc_pass': 129, 'arc_examples': 262, 'fail_reasons': '{"candidate_count=0": 134}', 'lowering_plan': 'constant D4 orbit masks plus small Equal/Where or tiny ScatterND', 'reject_risk': 'low if bbox size is static/small; reject if dynamic bbox requires full-grid GatherND'}

## 解釈

full pass hitが出たら次はONNX lowering。出ない場合は、P0 sparse/object tasksにはD4/rectangle/lineより強いobject-role grammarが必要。

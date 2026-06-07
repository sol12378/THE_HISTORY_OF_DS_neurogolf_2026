# exp_b010_component_object_role_sparse_fill_miner

## 目的

coordinate-firstでなく、object/component/color-role firstのsparse fill ruleを探索する。候補はrow/column reduction、directional reduction、small bbox mask、hole/component maskへloweringできるものに限定する。

## 結果

- target tasks: 9
- evaluated candidates: 2
- full pass hits: 2
- train/test pass hits: 0
- best partial: {'task_id': 37, 'candidate': 'opposite_ray_all_same_color', 'status': 'full_pass', 'train_pass': 3, 'train_examples': 3, 'test_pass': 1, 'test_examples': 1, 'arc_pass': 262, 'arc_examples': 262, 'total_pass': 266, 'total_examples': 266, 'avg_changed_cells': 8.466165413533835, 'fail_reasons': '{}', 'lowering_plan': 'row/column/diagonal reductions'}

## 判断

full passがあればtiny mask/reduction loweringへ進む。なければ、単一ruleでは不足なので、train-fitではなく部分一致からtask-specific decision treeを合成する。

# exp_b018_high_precision_island_or_tree_miner

## 目的

exp_b017で見つかったhigh precision feature islandをOR treeとして足し、train full fitする説明可能branch ruleを探索する。

## 結果

- target tasks: 5
- evaluated candidates: 4
- full pass hits: 0
- train/test hits: 0
- train-fit hits: 0
- rows: [{'task_id': 286, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 162199, 'gain_to_250': 6.475118337554084, 'status': 'partial', 'branch_count': 10, 'train_positive_cells': 92, 'train_negative_cells': 330, 'train_covered_positive': 10, 'train_false_positive': 0, 'train_pass': 0, 'train_examples': 2, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'total_pass': 0, 'total_examples': 265, 'branches': '[["ray_lr", "-1:6"], ["ray_lr", "-3:4"], ["ray_lr", "-5:2"], ["ray_lr", "2:-2"], ["ray_lr", "2:-5"], ["ray_lr", "4:-3"], ["ray_lr", "6:-1"], ["ray_ud", "-1:2"], ["ray_ud", "-2:2"], ["ray_ud", "2:0"]]', 'fail_reasons': '{"ok": 265}', 'lowering_plan': 'OR of ray/bbox/neighborhood feature-value masks; lower only if branch_count remains small'}, {'task_id': 133, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 193923, 'gain_to_250': 6.653755534178206, 'status': 'partial', 'branch_count': 3, 'train_positive_cells': 89, 'train_negative_cells': 1436, 'train_covered_positive': 6, 'train_false_positive': 0, 'train_pass': 0, 'train_examples': 4, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 262, 'total_pass': 0, 'total_examples': 267, 'branches': '[["ray_d1", "-4:-3"], ["ray_d1", "-5:-2"], ["ray_d1", "-6:-1"]]', 'fail_reasons': '{"ok": 258, "color_conflict": 9}', 'lowering_plan': 'OR of ray/bbox/neighborhood feature-value masks; lower only if branch_count remains small'}, {'task_id': 285, 'compiler_lane': 'L2_local_predicate_sparse_fill', 'strict_cost': 151996, 'gain_to_250': 6.410148565830427, 'status': 'partial', 'branch_count': 6, 'train_positive_cells': 81, 'train_negative_cells': 6001, 'train_covered_positive': 6, 'train_false_positive': 0, 'train_pass': 0, 'train_examples': 3, 'test_pass': 0, 'test_examples': 1, 'arc_pass': 0, 'arc_examples': 261, 'total_pass': 0, 'total_examples': 265, 'branches': '[["ray_d1", "-6:1"], ["ray_d2", "-3:1"], ["ray_lr", "1:-6"], ["ray_lr", "1:-7"], ["ray_lr", "2:-5"], ["ray_lr", "2:-6"]]', 'fail_reasons': '{"ok": 265}', 'lowering_plan': 'OR of ray/bbox/neighborhood feature-value masks; lower only if branch_count remains small'}]

## 判断

train-fitがあればbranch削減とfull loweringへ進む。なければpairwise feature conjunctionが必要。
